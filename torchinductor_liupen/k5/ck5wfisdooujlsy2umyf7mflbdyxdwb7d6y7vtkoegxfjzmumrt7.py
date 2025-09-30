
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_70', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 16384) % 128
    x1 = (xindex // 128) % 128
    x0 = xindex % 128
    x3 = (xindex // 2097152)
    tmp0 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp1 = (-1) + x2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = (-1) + x1
    tmp5 = tmp4 >= tmp2
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp3 & tmp5
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + ((-16257) + x0 + (127*x1) + (16129*x2) + (2048383*x3)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.load(in_ptr2 + ((-16257) + x0 + (127*x1) + (16129*x2) + (2048383*x3)), tmp9, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tmp15 + tmp21
    tmp23 = tmp0 + tmp22
    tl.store(in_out_ptr0 + (x4), tmp23, None)
