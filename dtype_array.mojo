from algorithm import vectorize
from benchmark import benchmark
from memory.reference import AddressSpace
from memory.unsafe import bitcast, DTypePointer
from memory.unsafe_pointer import UnsafePointer
from random import randint, seed
from sys.info import has_neon, simdwidthof
from sys.intrinsics import llvm_intrinsic
from utils.static_tuple import StaticTuple


@always_inline
fn _dtype_array_construction_checks[type: DType, size: Int]():
    constrained[type != DType.invalid, "type cannot be DType.invalid"]()
    constrained[
        type != DType.bfloat16 or not has_neon(),
        "DType.bf16 is not supported for ARM architectures",
    ]()
    constrained[size > 0, "size must be > 0"]()


fn _min(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


struct DTypeArray[
    type: DType, size: Int, address_space: AddressSpace = AddressSpace.GENERIC
]:
    alias _default_alignment = DTypePointer[
        type, address_space
    ]._default_alignment
    alias _default_width = simdwidthof[type]()
    alias _small_size = 64
    alias _small_storage_type = Optional[
        __mlir_type[
            `!pop.simd<`,
            _min(size, Self._small_size).value,
            `, `,
            type.value,
            `>`,
        ]
    ]

    var _small_storage: Self._small_storage_type
    var _storage_ptr: DTypePointer[type, address_space]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(inout self):
        _dtype_array_construction_checks[type, size]()

        self._storage_ptr = DTypePointer[type, address_space]()

        @parameter
        if Self._is_small():
            self._small_storage = Self._small_storage_type(
                __mlir_op.`kgen.undef`[_type = Self._small_storage_type.T]()
            )
            self._storage_ptr = self._get_small_storage_as_ptr()
        else:
            self._small_storage = Self._small_storage_type(None)
            self._storage_ptr = DTypePointer[type, address_space].alloc(size)

    @always_inline
    fn __init__(inout self, values: VariadicList[Scalar[type]]):
        self = DTypeArray[type, size, address_space]()

        for i in range(_min(size, len(values))):
            self._storage_ptr.store(i, values[i])

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        # print("copyinit DTypeArray")
        self = DTypeArray[type, size, address_space]()

        @parameter
        @always_inline
        fn copy_element[width: Int](i: Int):
            var element = existing._storage_ptr.load[
                width = Self._default_width, alignment = Self._default_alignment
            ](i)
            self._storage_ptr.store[
                width = Self._default_width, alignment = Self._default_alignment
            ](i, element)

        vectorize[copy_element, Self._default_width, size=size]()

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        # print("moveinit DTypeArray")

        @parameter
        if Self._is_small():
            self = existing
        else:
            self._small_storage = Self._small_storage_type(None)
            self._storage_ptr = existing._storage_ptr

    @always_inline
    fn __del__(owned self):
        @parameter
        if not Self._is_small():
            self._storage_ptr.free()

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __getitem__[index: Int](self) -> Scalar[type]:
        constrained[0 <= index < size, "index must be within bounds"]()
        return self._storage_ptr.load(index)

    @always_inline
    fn __setitem__[index: Int](inout self, value: Scalar[type]):
        constrained[0 <= index < size, "index must be within bounds"]()
        self._storage_ptr.store(index, value)

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn _is_small() -> Bool:
        return size <= Self._small_size

    @always_inline
    fn _get_small_storage_as_ptr(self) -> DTypePointer[type, address_space]:
        @parameter
        if Self._is_small():
            return DTypePointer[type, address_space](
                UnsafePointer.address_of(
                    self._small_storage.unsafe_value()
                ).bitcast[Scalar[type], address_space]()
            )
        else:
            return DTypePointer[type, address_space]()

    @always_inline
    fn unsafe_gather[
        width: Int, //, *, alignment: Int = Self._default_alignment
    ](
        self,
        index: SIMD[_, width],
        *,
        mask: SIMD[DType.bool, width],
        default: SIMD[type, width],
    ) -> SIMD[type, width]:
        @parameter
        if (type == DType.float32) and (size == 8) and (index.type == DType.int32):
            return llvm_intrinsic[
                "llvm.x86.avx2.gather.d.ps.256",
                __mlir_type[`!pop.simd<`, width.value, `, `, type.value, `>`],
                has_side_effect=False,
            ](
                default,
                self._storage_ptr,
                index,
                mask,
                Scalar[DType.int8](4),
            )
        else:
            return self._storage_ptr.gather[alignment=alignment](
                index, mask=mask, default=default
            )

    @always_inline
    fn unsafe_ptr(self) -> DTypePointer[type, address_space]:
        return self._storage_ptr


@always_inline
fn _check_float_table_constraints[size: Int, type: DType]() -> None:
    """Checks the constraints of the `FloatTable`."""
    constrained[size > 0, "size must be > 0"]()
    constrained[
        type.is_floating_point(), "dtype must be a floating-point type"
    ]()


@register_passable("trivial")
struct FloatTable[size: Int, type: DType]:
    var _data: StaticTuple[Scalar[type], size]

    @staticmethod
    fn from_value[value: Scalar[type]]() -> Self:
        _check_float_table_constraints[size, type]()

        var data = StaticTuple[Scalar[type], size]()

        @parameter
        for i in range(size):
            data[i] = value

        return Self {_data: data}

    @always_inline
    fn unsafe_lookup(self: Self, index: SIMD) -> SIMD[type, index.size]:
        constrained[
            index.type.is_integral(), "index must be an integral type"
        ]()
        var result = SIMD[type, index.size]()

        @parameter
        for i in range(index.size):
            result[i] = self._data[int(index[i])]

        return result


@value
struct FloatTable2[size: Int, type: DType]:
    var _data: DTypeArray[type, size]

    @staticmethod
    fn from_value[value: Scalar[type]]() -> Self:
        _check_float_table_constraints[size, type]()

        var data = DTypeArray[type, size]()

        @parameter
        for i in range(size):
            data.__setitem__[i](value)

        return Self(_data=data)

    @always_inline
    fn unsafe_lookup(self: Self, index: SIMD) -> SIMD[type, index.size]:
        constrained[
            index.type.is_integral(), "index must be an integral type"
        ]()
        alias nan: SIMD[type, index.size] = math.nan[type]()

        return self._data.unsafe_gather(index, mask=True, default=nan)


fn gather[
    type: DType, size: Int
](
    base: DTypePointer[type, AddressSpace.GENERIC],
    index: SIMD[DType.int32, size],
    mask: SIMD[type, size],
) -> SIMD[type, size]:
    var undef = SIMD[type, size](-1.0)

    @parameter
    if (type == DType.float32) and (size == 8):
        return llvm_intrinsic[
            "llvm.x86.avx2.gather.d.ps.256",
            __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`],
            has_side_effect=False,
        ](
            undef,
            base,
            index,
            mask,
            Scalar[DType.int8](2),
        )
    else:
        return undef


alias INTEGRAL_TYPE = DType.int32
alias FLOAT_TYPE = DType.float32
alias SIZE = 1024
alias FLOAT_TABLE = FloatTable2[SIZE, FLOAT_TYPE].from_value[0]()


fn math_func[
    width: Int = simdwidthof[FLOAT_TYPE]()
]() -> SIMD[FLOAT_TYPE, width]:
    var index = SIMD[INTEGRAL_TYPE, width]()
    var index_ptr = DTypePointer[INTEGRAL_TYPE](
        UnsafePointer.address_of(index).bitcast[Scalar[INTEGRAL_TYPE]]()
    )
    randint(index_ptr, width, 0, SIZE - 1)
    return FLOAT_TABLE.unsafe_lookup(index)


fn main():
    var raw_data = SIMD[DType.float16, 16](0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    var data = bitcast[DType.float32, 8](raw_data)
    var ptr = DTypePointer[data.type](
        UnsafePointer.address_of(data).bitcast[Scalar[data.type]]()
    )
    var index = SIMD[DType.int32, 8](7, 4, 5, 6, 3, 0, 1, 2)
    var mask = SIMD[DType.int32, 8](0xffffffffe0000000)
    print(bitcast[DType.float32, 8](mask))
    var res = gather(ptr, index, bitcast[DType.float32, 8](mask))
    print(bitcast[DType.float16, 16](res))
