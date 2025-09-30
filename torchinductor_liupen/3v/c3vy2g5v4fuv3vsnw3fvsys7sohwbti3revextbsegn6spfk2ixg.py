
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/s7/cs7uxx3dqusugqjdgwvekcqxijyyhxvu7j4xborfeqcekwisfr57.py
# Source Nodes: [x], Original ATen: [aten._to_copy]
# x => convert_element_type_1
triton_poi_fused__to_copy_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/mb/cmba236gf6yhihsxunvojvne5goxhcgmjdo3zjhxrfgvf6zhfa6t.py
# Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
# x => convert_element_type, convert_element_type_1, convolution
triton_poi_fused__to_copy_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32, 2097152], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 2097152
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x1)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x1 + (2097152*y0)), tmp3, ymask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/zi/czi24gowysd32iepeofrmo4jw5bgk436nmxky55rpaqix6frexz2.py
# Source Nodes: [conv3d], Original ATen: [aten._to_copy]
# conv3d => convert_element_type_3
triton_poi_fused__to_copy_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gn/cgnipiihh5bdyx5q3w6yuzcqyh4ocgujozc2f3iv2gnbqmoiuc6o.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_red_fused_native_group_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 262144],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 209716
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (209716*x0)
        tmp1 = tl.full([1, 1], 2097152, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((2097152*x1) + ((r2 + (209716*x0)) % 2097152)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/pk/cpk3jrhjy3pbszftlyh4imywjs6e5hcxjmqwfb2xdhkacwejpmcp.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_per_fused_native_group_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gn/cgnnyevcatuwdszuephpd4hvsxupplguoq5bnzqqsv3ggggcw3fr.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_7
# group_norm => add_1, mul_1
triton_poi_fused__to_copy_native_group_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 2097152)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 2097152.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/tn/ctnhav6l6mtmerwdwqepstafpwdylisbrsnfvtef47qh5473awfb.py
# Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
# conv3d_1 => convert_element_type_6
triton_poi_fused__to_copy_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/jo/cjoy5wavcd7hmpajrdrnvlkvllozoojbcpwk3ckfrhktyyxrwwyh.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_5, convert_element_type_6, convert_element_type_7, convolution_2
# gelu => add_2, convert_element_type_8, convert_element_type_9, erf, mul_2, mul_3, mul_4
# group_norm => add_1, mul_1
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2097152)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/v6/cv6oo7w4ltrfbncmtglesqdfsxbyq56uwwanpoj3b6uiccucauiw.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_3
# conv3d_1 => convert_element_type_5, convert_element_type_6, convert_element_type_7, convolution_2
# conv3d_2 => convert_element_type_10, convert_element_type_11, convolution_3
# gelu => add_2, convert_element_type_8, convert_element_type_9, erf, mul_2, mul_3, mul_4
# group_norm => add_1, mul_1
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2097152)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/y2/cy2ornnwzmyacetwxurroq6r3nvapvrabnt3adhvsnxt6oia4tqu.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_7
# conv3d_1 => convert_element_type_15, convert_element_type_16, convert_element_type_17, convolution_5
# conv3d_2 => convert_element_type_20, convert_element_type_21, convolution_6
# gelu => add_6, convert_element_type_18, convert_element_type_19, erf_1, mul_7, mul_8, mul_9
# group_norm => add_5, mul_6
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2097152)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/uy/cuywfxe6xnijijoetykvcqr7m2yl6lahpmg5hxowi3erfolge5ev.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_2
triton_red_fused_native_group_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*((r2 + (32768*x0)) // 4096)) + (262144*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gb/cgb7avt5mharojqspkjaqb2zvppbjqpnnedbg6oqd7xzpqwkcujy.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_2
triton_per_fused_native_group_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (8*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (8*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (8*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/qy/cqyw2m3yubac664b3l246tosyuksi4ubydhbh6yac4cuc4oizkba.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_27
# group_norm => add_9, mul_11
triton_poi_fused__to_copy_native_group_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 262144.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/3t/c3tl5johkwf64ldkl643dmweuxry74z2getmki4pdbewthl62kkv.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_25, convert_element_type_26, convert_element_type_27, convolution_8
# gelu => add_10, convert_element_type_28, convert_element_type_29, erf_2, mul_12, mul_13, mul_14
# group_norm => add_9, mul_11
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/pd/cpdzqxkgkq4efqs6t2k3p7jpcgrzmbynuzolievo5zj33f3x466p.py
# Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
# conv3d_2 => convert_element_type_31
triton_poi_fused__to_copy_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/na/cnalffh7ibjkb57gk7j7e4jlhvdeytdv52eqviqytaruwbpdh3cq.py
# Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_11
# conv3d_1 => convert_element_type_25, convert_element_type_26, convert_element_type_27, convolution_8
# conv3d_2 => convert_element_type_30, convert_element_type_31, convolution_9
# conv3d_3 => convert_element_type_32, convert_element_type_33, convolution_10
# gelu => add_10, convert_element_type_28, convert_element_type_29, erf_2, mul_12, mul_13, mul_14
# group_norm => add_9, mul_11
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/mz/cmzpduolsxax6pnk5ihyubqrkfup5z7yiejwthgrkovhic6472zx.py
# Source Nodes: [conv3d], Original ATen: [aten._to_copy]
# conv3d => convert_element_type_35
triton_poi_fused__to_copy_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/4y/c4yukr5no3oh7kh6egazb25xptfz3qo7opgtobswlwq3lfo5jyin.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_3
triton_red_fused_native_group_norm_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 52429
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (52429*x0)
        tmp1 = tl.full([1, 1], 262144, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((262144*x1) + ((r2 + (52429*x0)) % 262144)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/xv/cxvt7kqrizt75kdegklbhglqp45l2knbkuihxy5wbxyvts27pxpy.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_3
triton_per_fused_native_group_norm_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/6m/c6mfgd22c2dstupq5ldi7e43r73hfrdduptju4qlfn3ip7bqw4nm.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_39
# group_norm => add_13, mul_16
triton_poi_fused__to_copy_native_group_norm_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 262144.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gz/cgz57raz35fzvrlolx2ntnswpxe6mlej3r5w76qbraw5mpxgtsmq.py
# Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
# conv3d_1 => convert_element_type_38
triton_poi_fused__to_copy_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/vi/cviuoulpsa67paocapqyad2wespsy7aaq7u7ooyqj3bfsur5j4hd.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_37, convert_element_type_38, convert_element_type_39, convolution_12
# gelu => add_14, convert_element_type_40, convert_element_type_41, erf_3, mul_17, mul_18, mul_19
# group_norm => add_13, mul_16
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/bb/cbbd556ka5ycccaaukrccezfxcfi47vs6uprc6oh6swhyrehdoup.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_15
# conv3d_1 => convert_element_type_37, convert_element_type_38, convert_element_type_39, convolution_12
# conv3d_2 => convert_element_type_42, convert_element_type_43, convolution_13
# gelu => add_14, convert_element_type_40, convert_element_type_41, erf_3, mul_17, mul_18, mul_19
# group_norm => add_13, mul_16
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/tf/ctf2xtxnh6676r4mnsppnrmsb7kv325zzweya4xegu2wvfvhcqdi.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_5
triton_red_fused_native_group_norm_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((1024*((r2 + (8192*x0)) // 1024)) + (32768*x1) + (r2 % 1024)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/eq/ceqkhcrcoq5voa5yg2o6syo7h7peq3tt352efvodrqwhpvgxkn3g.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_5
triton_per_fused_native_group_norm_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/uk/cukfrfdp425rzkoqkrrp5vknbfq2qsi5b66hqb3ps3rmp36hnli5.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_59
# group_norm => add_21, mul_26
triton_poi_fused__to_copy_native_group_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_25', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 32768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/42/c42c3zphsqfiuilyffvnk26e7q2tw7pd4o4okg7t3ltbwh5ske46.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_57, convert_element_type_58, convert_element_type_59, convolution_18
# gelu => add_22, convert_element_type_60, convert_element_type_61, erf_5, mul_27, mul_28, mul_29
# group_norm => add_21, mul_26
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/5m/c5malx7f6od6solntqpykwwomchmyhxfa5yl5fd4q42uodqlbm7s.py
# Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
# conv3d_2 => convert_element_type_63
triton_poi_fused__to_copy_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/dl/cdllvpacflmiz7o5djtxfh46jxdqbn6cth6jtmqnz2zzxj7i67ri.py
# Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_23
# conv3d_1 => convert_element_type_57, convert_element_type_58, convert_element_type_59, convolution_18
# conv3d_2 => convert_element_type_62, convert_element_type_63, convolution_19
# conv3d_3 => convert_element_type_64, convert_element_type_65, convolution_20
# gelu => add_22, convert_element_type_60, convert_element_type_61, erf_5, mul_27, mul_28, mul_29
# group_norm => add_21, mul_26
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_28', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/lh/clhh6mwg5icmaeium5dqomlmpxvd5pliuvtbj4tdbzcce5ywlb5n.py
# Source Nodes: [conv3d], Original ATen: [aten._to_copy]
# conv3d => convert_element_type_67
triton_poi_fused__to_copy_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/bj/cbjqhjryazvuh53rdlx7czxubnm6vjbcdbzyqmy3gf6ol3jgnkq4.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_6
triton_red_fused_native_group_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((1024*((r2 + (8192*x0)) // 1024)) + (32768*x1) + (r2 % 1024)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/yh/cyhrdg3lv7ovtl3l5s67tp43rqmv4ggs2w643ap26fjs767ppqxe.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_6
triton_per_fused_native_group_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/bq/cbqsswmhla2n3luwtqgbgduon34jf3jpa323i6iwrskiu7kpmsf7.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_71
# group_norm => add_25, mul_31
triton_poi_fused__to_copy_native_group_norm_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_32', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 32768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/un/cunge6blpevkn3h5qypbf7qlpx7taddikcy6i4immttfa2jxcmjn.py
# Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
# conv3d_1 => convert_element_type_70
triton_poi_fused__to_copy_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/fh/cfhs4e3bptwxbqycifvzlbyrdz6vc46hosakqjeduwjbdbvuujhx.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_69, convert_element_type_70, convert_element_type_71, convolution_22
# gelu => add_26, convert_element_type_72, convert_element_type_73, erf_6, mul_32, mul_33, mul_34
# group_norm => add_25, mul_31
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/5k/c5kpjhdnocagmj5t6ve6el4b6xzqpn6zf37og77lafc7kylyydan.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_27
# conv3d_1 => convert_element_type_69, convert_element_type_70, convert_element_type_71, convolution_22
# conv3d_2 => convert_element_type_74, convert_element_type_75, convolution_23
# gelu => add_26, convert_element_type_72, convert_element_type_73, erf_6, mul_32, mul_33, mul_34
# group_norm => add_25, mul_31
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 32768)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gz/cgzcekazhywuenuzer6maicnwrdt4wxnru5iyz4zztncqeqoldxv.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_31
# conv3d_1 => convert_element_type_79, convert_element_type_80, convert_element_type_81, convolution_25
# conv3d_2 => convert_element_type_84, convert_element_type_85, convolution_26
# gelu => add_30, convert_element_type_82, convert_element_type_83, erf_7, mul_37, mul_38, mul_39
# group_norm => add_29, mul_36
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_36', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/lm/clmivyl6gfzjqlvxgsjdn6borsjogr6o52bvb5buisfhs2rlogfp.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_91
# group_norm => add_33, mul_41, var_mean_8
triton_red_fused__to_copy_native_group_norm_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[128, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_group_norm_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp1.to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = 4096.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/za/czazrcgx3wrgvgl6ddzr4axvu2whunax5mtr2owkknul5ffkhzc6.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_89, convert_element_type_90, convert_element_type_91, convolution_28
# gelu => add_34, convert_element_type_92, convert_element_type_93, erf_8, mul_42, mul_43, mul_44
# group_norm => add_33, mul_41
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/kx/ckxkt3ofmy7rwediyx2f26ndv3ftofet4ncofwry2ddlufgjt5qz.py
# Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
# conv3d_2 => convert_element_type_95
triton_poi_fused__to_copy_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_39', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/mu/cmu7fwxvwq3phtgj6qrznspxvvs4w4ymnokajyittru6u475a5rs.py
# Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_35
# conv3d_1 => convert_element_type_89, convert_element_type_90, convert_element_type_91, convolution_28
# conv3d_2 => convert_element_type_94, convert_element_type_95, convolution_29
# conv3d_3 => convert_element_type_96, convert_element_type_97, convolution_30
# gelu => add_34, convert_element_type_92, convert_element_type_93, erf_8, mul_42, mul_43, mul_44
# group_norm => add_33, mul_41
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/mz/cmzeebrzh6svo23vebnm44bs7gwloqzepx4idxsv5rwx3zibgnah.py
# Source Nodes: [conv3d], Original ATen: [aten._to_copy]
# conv3d => convert_element_type_99
triton_poi_fused__to_copy_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_41', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/ho/cholnhjpzmdo24tglhwyiedjornp4cdaawxwtebippeqc2ylgnr5.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_103
# group_norm => add_37, mul_46, var_mean_9
triton_red_fused__to_copy_native_group_norm_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_group_norm_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp1.to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = 4096.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/qn/cqnvirwkqtlmwpihmdwqj457h3bj2o42szxt4qz7a562hma4wf6y.py
# Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
# conv3d_1 => convert_element_type_102
triton_poi_fused__to_copy_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_43', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/pn/cpnc2buktmi6hxozhedjww7wnm6zbxn3jbdlfap32uzf3imde37c.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_101, convert_element_type_102, convert_element_type_103, convolution_32
# gelu => add_38, convert_element_type_104, convert_element_type_105, erf_9, mul_47, mul_48, mul_49
# group_norm => add_37, mul_46
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gf/cgfbtwrykduky5ez7f7hy7qim2vkugfgbmzqn4u2pij74xq4y7lm.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_39
# conv3d_1 => convert_element_type_101, convert_element_type_102, convert_element_type_103, convolution_32
# conv3d_2 => convert_element_type_106, convert_element_type_107, convolution_33
# gelu => add_38, convert_element_type_104, convert_element_type_105, erf_9, mul_47, mul_48, mul_49
# group_norm => add_37, mul_46
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/3b/c3bvkelqn6wzurnvtwm2a5ziz4q2abav5firnsgu446n4l7ska6y.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_123
# group_norm => add_45, mul_56, var_mean_11
triton_per_fused__to_copy_native_group_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_group_norm_46', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/ft/cftoaknz3wy6csy5b53bl3tetaecynuiijszd4tlc7oewb5ubpv4.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_121, convert_element_type_122, convert_element_type_123, convolution_38
# gelu => add_46, convert_element_type_124, convert_element_type_125, erf_11, mul_57, mul_58, mul_59
# group_norm => add_45, mul_56
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_47', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/35/c357cwafc5qjwkustarm5xmtwkhzs7aknfy2idfrqwybuqyktnvy.py
# Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
# conv3d_2 => convert_element_type_127
triton_poi_fused__to_copy_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/dc/cdc5n3loeaqjj7wxbzf6hgzh2tllxrxjrbiz23l7lhm46dlocv2w.py
# Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_47
# conv3d_1 => convert_element_type_121, convert_element_type_122, convert_element_type_123, convolution_38
# conv3d_2 => convert_element_type_126, convert_element_type_127, convolution_39
# conv3d_3 => convert_element_type_128, convert_element_type_129, convolution_40
# gelu => add_46, convert_element_type_124, convert_element_type_125, erf_11, mul_57, mul_58, mul_59
# group_norm => add_45, mul_56
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_49', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/x4/cx45weasf45b76kdoeutbigfcauqwd5fua53ytxqekx6ow46dksv.py
# Source Nodes: [conv3d], Original ATen: [aten._to_copy]
# conv3d => convert_element_type_131
triton_poi_fused__to_copy_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_50', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/e7/ce7yhfvknwbsxwogzvucejn3hhiifotngm7tr57mfh3n2bkxkrp4.py
# Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d_1 => convert_element_type_135
# group_norm => add_49, mul_61, var_mean_12
triton_per_fused__to_copy_native_group_norm_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_group_norm_51', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/6n/c6nlzk3oifwmu74ylypufuh43lk7v2wpvye5ap2v46uuxm3nj6zs.py
# Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
# conv3d_1 => convert_element_type_134
triton_poi_fused__to_copy_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_52', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/6v/c6vdxptpawju4f4kdo5hlcch45ooe76pb3d5nvog43pa5tccurdh.py
# Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d_1 => convert_element_type_133, convert_element_type_134, convert_element_type_135, convolution_42
# gelu => add_50, convert_element_type_136, convert_element_type_137, erf_12, mul_62, mul_63, mul_64
# group_norm => add_49, mul_61
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/3z/c3z4jqzrdswssu7ipnevrj7gsxgomwnuusitt44kdzsp3ggb3uzx.py
# Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_51
# conv3d_1 => convert_element_type_133, convert_element_type_134, convert_element_type_135, convolution_42
# conv3d_2 => convert_element_type_138, convert_element_type_139, convolution_43
# gelu => add_50, convert_element_type_136, convert_element_type_137, erf_12, mul_62, mul_63, mul_64
# group_norm => add_49, mul_61
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_54', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/io/ciotqabe3phbils5ok5wr2jd5b65ugqq4yv55kmz2sn2dfa3pssc.py
# Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d => convert_element_type_155
# group_norm => add_57, mul_71, var_mean_14
triton_red_fused__to_copy_native_group_norm_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_group_norm_55', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3375
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3375*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (3375*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp1.to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = 3375.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (3375*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/rm/crmkovfryctt62nnjlxy7ayrjmh6dxnbbiv44kwhsbigpmdz47nq.py
# Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d => convert_element_type_153, convert_element_type_154, convert_element_type_155, convolution_48
# gelu => add_58, convert_element_type_156, convert_element_type_157, erf_14, mul_72, mul_73, mul_74
# group_norm => add_57, mul_71
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_56', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3456000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3375)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/dy/cdyrrdqul6wqpuhmeirab6yahr7r2ocj36i2dsqsoc3nyfdjpqyh.py
# Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_59
# conv3d => convert_element_type_153, convert_element_type_154, convert_element_type_155, convolution_48
# conv3d_1 => convert_element_type_158, convert_element_type_159, convolution_49
# conv_transpose3d_1 => convert_element_type_160, convert_element_type_161, convolution_50
# dec_x => add_60
# gelu => add_58, convert_element_type_156, convert_element_type_157, erf_14, mul_72, mul_73, mul_74
# group_norm => add_57, mul_71
# pad => constant_pad_nd
# pad_1 => constant_pad_nd_1
triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_57', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 256) % 16
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x3 = (xindex // 4096)
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
    tmp10 = tl.load(in_ptr0 + ((-241) + x0 + (15*x1) + (225*x2) + (3375*x3)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.load(in_ptr2 + ((-241) + x0 + (15*x1) + (225*x2) + (3375*x3)), tmp9, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tmp15 + tmp21
    tmp23 = tmp0 + tmp22
    tl.store(in_out_ptr0 + (x4), tmp23, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/3g/c3g43gowkiaswnjwgc44t6hybynq4aumlysjfkcwtmxhhdi4z6xp.py
# Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d => convert_element_type_187
# group_norm => add_70, mul_86, var_mean_17
triton_red_fused__to_copy_native_group_norm_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_group_norm_58', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 29791
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (29791*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (29791*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp1.to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = 29791.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (29791*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/6q/c6q4zimu4wiksm7ppy6jmhgg7aghk5grujxgi3aagsrlblqjpytz.py
# Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d => convert_element_type_185, convert_element_type_186, convert_element_type_187, convolution_58
# gelu => add_71, convert_element_type_188, convert_element_type_189, erf_17, mul_87, mul_88, mul_89
# group_norm => add_70, mul_86
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_59', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15252992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 29791)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/gg/cgg5j6e2ofvqmvqshyhljhnzwsk7ikq3vikoryuidxgx4lju3zio.py
# Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_1, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_72
# conv3d => convert_element_type_185, convert_element_type_186, convert_element_type_187, convolution_58
# conv3d_1 => convert_element_type_190, convert_element_type_191, convolution_59
# conv_transpose3d_1 => convert_element_type_192, convert_element_type_193, convolution_60
# dec_x_1 => add_73
# gelu => add_71, convert_element_type_188, convert_element_type_189, erf_17, mul_87, mul_88, mul_89
# group_norm => add_70, mul_86
# pad => constant_pad_nd_2
# pad_1 => constant_pad_nd_3
triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_60', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1024) % 32
    x1 = (xindex // 32) % 32
    x0 = xindex % 32
    x3 = (xindex // 32768)
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
    tmp10 = tl.load(in_ptr0 + ((-993) + x0 + (31*x1) + (961*x2) + (29791*x3)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.load(in_ptr2 + ((-993) + x0 + (31*x1) + (961*x2) + (29791*x3)), tmp9, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tmp15 + tmp21
    tmp23 = tmp0 + tmp22
    tl.store(in_out_ptr0 + (x4), tmp23, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/xa/cxat5aytwhkojvmuqbxocdrwedd5y7ff276cg7pyb5zrv5gudsn4.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_20
triton_red_fused_native_group_norm_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_61', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 83349
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3969*((r2 + (83349*x0)) // 3969)) + (250047*x1) + (r2 % 3969)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/zm/czmcmksvuqh4qyp2d7dwdqoxq7t3uv5qwcoy27e6324dj45bf4zq.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_20
triton_per_fused_native_group_norm_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_62', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/tu/ctukwx2zvann7sm3jgg74g4wlaooy4mismgyxahk2krfs2glqrci.py
# Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d => convert_element_type_219
# group_norm => add_83, mul_101
triton_poi_fused__to_copy_native_group_norm_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_63', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32006016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 250047)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 250047.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/52/c52ejaouhzf3dyd7ggaos45qtlp54st4vk6fiyqqn2vqe3hxlf2u.py
# Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d => convert_element_type_217, convert_element_type_218, convert_element_type_219, convolution_68
# gelu => add_84, convert_element_type_220, convert_element_type_221, erf_20, mul_102, mul_103, mul_104
# group_norm => add_83, mul_101
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_64', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64012032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 250047)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/z6/cz6555nrztryn7c2bjzg4ayqo2wsnrhyxecjfklbg2seagitl5e2.py
# Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_2, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_85
# conv3d => convert_element_type_217, convert_element_type_218, convert_element_type_219, convolution_68
# conv3d_1 => convert_element_type_222, convert_element_type_223, convolution_69
# conv_transpose3d_1 => convert_element_type_224, convert_element_type_225, convolution_70
# dec_x_2 => add_86
# gelu => add_84, convert_element_type_220, convert_element_type_221, erf_20, mul_102, mul_103, mul_104
# group_norm => add_83, mul_101
# pad => constant_pad_nd_4
# pad_1 => constant_pad_nd_5
triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_65', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 4096) % 64
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 262144)
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
    tmp10 = tl.load(in_ptr0 + ((-4033) + x0 + (63*x1) + (3969*x2) + (250047*x3)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.load(in_ptr2 + ((-4033) + x0 + (63*x1) + (3969*x2) + (250047*x3)), tmp9, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tmp15 + tmp21
    tmp23 = tmp0 + tmp22
    tl.store(in_out_ptr0 + (x4), tmp23, None)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/mk/cmkfum3ro37y6n6pejffvcmt6t2wfgieggbhp4dmwuoexkwprna6.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_23
triton_red_fused_native_group_norm_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_66', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128024*x0)
        tmp1 = tl.full([1, 1], 2048383, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((2048383*x1) + ((r2 + (128024*x0)) % 2048383)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_combine(
            tmp19_mean, tmp19_m2, tmp19_weight,
            tmp16, tmp17, tmp18
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/ai/caihcwvzwvp6vdnzimarpbwmhgvlermzniwm2agncfjt4bruh475.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean_23
triton_per_fused_native_group_norm_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_67', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/qv/cqvrnzzs56njzacp7el6zxjhtuqnlq4ysmagrhwphfw2wh7hpe3w.py
# Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
# conv3d => convert_element_type_251
# group_norm => add_96, mul_116
triton_poi_fused__to_copy_native_group_norm_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_group_norm_68', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131096512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 2048383)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 2048383.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/sj/csjzrxuse3zxrw7oh7rrmwhepbfvhvictl7xkopatr3gx2ypjwm3.py
# Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
# conv3d => convert_element_type_249, convert_element_type_250, convert_element_type_251, convolution_78
# gelu => add_97, convert_element_type_252, convert_element_type_253, erf_23, mul_117, mul_118, mul_119
# group_norm => add_96, mul_116
triton_poi_fused__to_copy_convolution_gelu_native_group_norm_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[268435456], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_native_group_norm_69', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262193024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2048383)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/k5/ck5wfisdooujlsy2umyf7mflbdyxdwb7d6y7vtkoegxfjzmumrt7.py
# Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_3, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_98
# conv3d => convert_element_type_249, convert_element_type_250, convert_element_type_251, convolution_78
# conv3d_1 => convert_element_type_254, convert_element_type_255, convolution_79
# conv_transpose3d_1 => convert_element_type_256, convert_element_type_257, convolution_80
# dec_x_3 => add_99
# gelu => add_97, convert_element_type_252, convert_element_type_253, erf_23, mul_117, mul_118, mul_119
# group_norm => add_96, mul_116
# pad => constant_pad_nd_6
# pad_1 => constant_pad_nd_7
triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_70 = async_compile.triton('triton_', '''
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
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/xc/cxcehktwfnwzq2vo37rchmeu2u4wqlbegsnggixfyapdgfddlehr.py
# Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
# conv_transpose3d => convert_element_type_279
triton_poi_fused__to_copy_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_71', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /projects/weilab/liupeng/banis/torchinductor_liupen/ij/cijr7isrw7lbeyaa3muggvuruqi2xf4qkxnvvxp5jxi53a4cjhkn.py
# Source Nodes: [add, conv3d_1, conv3d_2, conv_transpose3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
# add => add_107
# conv3d_1 => convert_element_type_271, convert_element_type_272, convert_element_type_273, convolution_85
# conv3d_2 => convert_element_type_276, convert_element_type_277, convolution_86
# conv_transpose3d => convert_element_type_278, convert_element_type_279, convolution_87
# gelu => add_106, convert_element_type_274, convert_element_type_275, erf_25, mul_127, mul_128, mul_129
# group_norm => add_105, mul_126
triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8, 2097152], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_72', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '57D8074E0528F5E502FE74C16FB05E76D6DE5A9D1AA751AA36F98F05E31DE393', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7
    xnumel = 2097152
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + (2097152*y0)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (y0 + (7*x1)), tmp3, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (256, 1024, 1, 1, 1), (1024, 1, 1, 1, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg137_1, (256, ), (1, ))
    assert_size_stride(arg138_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (128, 512, 1, 1, 1), (512, 1, 1, 1, 1))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (64, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg179_1, (64, ), (1, ))
    assert_size_stride(arg180_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg181_1, (64, ), (1, ))
    assert_size_stride(arg182_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (64, ), (1, ))
    assert_size_stride(arg186_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg189_1, (64, ), (1, ))
    assert_size_stride(arg190_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg191_1, (64, ), (1, ))
    assert_size_stride(arg192_1, (64, ), (1, ))
    assert_size_stride(arg193_1, (64, ), (1, ))
    assert_size_stride(arg194_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg197_1, (64, ), (1, ))
    assert_size_stride(arg198_1, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg199_1, (64, ), (1, ))
    assert_size_stride(arg200_1, (64, ), (1, ))
    assert_size_stride(arg201_1, (64, ), (1, ))
    assert_size_stride(arg202_1, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (32, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg205_1, (32, ), (1, ))
    assert_size_stride(arg206_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg207_1, (32, ), (1, ))
    assert_size_stride(arg208_1, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg209_1, (32, ), (1, ))
    assert_size_stride(arg210_1, (32, ), (1, ))
    assert_size_stride(arg211_1, (32, ), (1, ))
    assert_size_stride(arg212_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg213_1, (64, ), (1, ))
    assert_size_stride(arg214_1, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg215_1, (32, ), (1, ))
    assert_size_stride(arg216_1, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(arg217_1, (32, ), (1, ))
    assert_size_stride(arg218_1, (32, ), (1, ))
    assert_size_stride(arg219_1, (32, ), (1, ))
    assert_size_stride(arg220_1, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(arg221_1, (64, ), (1, ))
    assert_size_stride(arg222_1, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(arg223_1, (32, ), (1, ))
    assert_size_stride(arg224_1, (32, 7, 1, 1, 1), (7, 1, 1, 1, 1))
    assert_size_stride(arg225_1, (7, ), (1, ))
    assert_size_stride(arg226_1, (32, 1, 1, 1, 1), (1, 1, 1, 1, 1))
    assert_size_stride(arg227_1, (32, ), (1, ))
    assert_size_stride(arg228_1, (1, 1, 128, 128, 128), (2097152, 1, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 1, 1, 1), (1, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [x], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg226_1, buf0, 32, grid=grid(32), stream=stream0)
        del arg226_1
        # Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
        buf1 = extern_kernels.convolution(arg228_1, buf0, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 32, 128, 128, 128), (67108864, 1, 524288, 4096, 32))
        del arg228_1
        del buf0
        buf2 = empty_strided_cuda((1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1), torch.float16)
        # Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
        triton_poi_fused__to_copy_convolution_1.run(buf1, arg227_1, buf2, 32, 2097152, grid=grid(32, 2097152), stream=stream0)
        del arg227_1
        del buf1
        buf3 = empty_strided_cuda((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), torch.float16)
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(arg0_1, buf3, 864, grid=grid(864), stream=stream0)
        del arg0_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        buf5 = empty_strided_cuda((1, 32, 1, 1, 10), (320, 10, 320, 320, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 32, 1, 1, 10), (320, 10, 320, 320, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 32, 1, 1, 10), (320, 10, 320, 320, 1), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_3.run(buf4, arg1_1, buf5, buf6, buf7, 320, 209716, grid=grid(320), stream=stream0)
        buf8 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf9 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf5, buf6, buf7, buf8, buf9, 32, 10, grid=grid(32), stream=stream0)
        buf11 = buf4; del buf4  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_5.run(buf11, arg1_1, buf8, buf9, arg2_1, arg3_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        buf12 = empty_strided_cuda((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg4_1, buf12, 2048, grid=grid(2048), stream=stream0)
        del arg4_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (1, 64, 128, 128, 128), (134217728, 2097152, 16384, 128, 1))
        del buf11
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7.run(buf14, arg5_1, 134217728, grid=grid(134217728), stream=stream0)
        del arg5_1
        buf15 = reinterpret_tensor(buf12, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf12  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg6_1, buf15, 2048, grid=grid(2048), stream=stream0)
        del arg6_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        del buf14
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_8.run(buf17, buf2, arg7_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg7_1
        del buf2
        buf18 = buf3; del buf3  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(arg8_1, buf18, 864, grid=grid(864), stream=stream0)
        del arg8_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf19, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        buf20 = buf7; del buf7  # reuse
        buf21 = buf6; del buf6  # reuse
        buf22 = buf5; del buf5  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_3.run(buf19, arg9_1, buf20, buf21, buf22, 320, 209716, grid=grid(320), stream=stream0)
        buf23 = buf9; del buf9  # reuse
        buf24 = buf8; del buf8  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf20, buf21, buf22, buf23, buf24, 32, 10, grid=grid(32), stream=stream0)
        buf26 = buf19; del buf19  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_5.run(buf26, arg9_1, buf23, buf24, arg10_1, arg11_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg10_1
        del arg11_1
        del arg9_1
        buf27 = reinterpret_tensor(buf15, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1), 0); del buf15  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg12_1, buf27, 2048, grid=grid(2048), stream=stream0)
        del arg12_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (1, 64, 128, 128, 128), (134217728, 2097152, 16384, 128, 1))
        del buf26
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7.run(buf29, arg13_1, 134217728, grid=grid(134217728), stream=stream0)
        del arg13_1
        buf30 = reinterpret_tensor(buf27, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf27  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg14_1, buf30, 2048, grid=grid(2048), stream=stream0)
        del arg14_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf31 = extern_kernels.convolution(buf29, buf30, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        del buf29
        buf32 = buf17; del buf17  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_9.run(buf32, buf31, arg15_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg15_1
        del buf31
        buf33 = buf18; del buf18  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(arg16_1, buf33, 864, grid=grid(864), stream=stream0)
        del arg16_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf34, (1, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf35 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf36 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf37 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_10.run(buf34, arg17_1, buf35, buf36, buf37, 256, 32768, grid=grid(256), stream=stream0)
        buf38 = buf24; del buf24  # reuse
        buf39 = buf23; del buf23  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_11.run(buf35, buf36, buf37, buf38, buf39, 32, 8, grid=grid(32), stream=stream0)
        buf41 = buf34; del buf34  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_12.run(buf41, arg17_1, buf38, buf39, arg18_1, arg19_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf42 = reinterpret_tensor(buf30, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1), 0); del buf30  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg20_1, buf42, 2048, grid=grid(2048), stream=stream0)
        del arg20_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf41
        buf44 = buf43; del buf43  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_13.run(buf44, arg21_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg21_1
        buf45 = empty_strided_cuda((64, 64, 1, 1, 1), (64, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_14.run(arg22_1, buf45, 4096, grid=grid(4096), stream=stream0)
        del arg22_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf44
        buf47 = buf42; del buf42  # reuse
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg24_1, buf47, 2048, grid=grid(2048), stream=stream0)
        del arg24_1
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy, aten.convolution]
        buf48 = extern_kernels.convolution(buf32, buf47, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        buf49 = buf46; del buf46  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_15.run(buf49, arg23_1, buf48, arg25_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg23_1
        del arg25_1
        del buf48
        buf50 = empty_strided_cuda((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), torch.float16)
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg26_1, buf50, 1728, grid=grid(1728), stream=stream0)
        del arg26_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf51 = extern_kernels.convolution(buf49, buf50, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf51, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        buf52 = reinterpret_tensor(buf22, (1, 64, 1, 1, 5), (320, 5, 320, 320, 1), 0); del buf22  # reuse
        buf53 = reinterpret_tensor(buf21, (1, 64, 1, 1, 5), (320, 5, 320, 320, 1), 0); del buf21  # reuse
        buf54 = reinterpret_tensor(buf20, (1, 64, 1, 1, 5), (320, 5, 320, 320, 1), 0); del buf20  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_17.run(buf51, arg27_1, buf52, buf53, buf54, 320, 52429, grid=grid(320), stream=stream0)
        buf55 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf56 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf52, buf53, buf54, buf55, buf56, 64, 5, grid=grid(64), stream=stream0)
        buf58 = buf51; del buf51  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_19.run(buf58, arg27_1, buf55, buf56, arg28_1, arg29_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        buf59 = empty_strided_cuda((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg30_1, buf59, 8192, grid=grid(8192), stream=stream0)
        del arg30_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf60 = extern_kernels.convolution(buf58, buf59, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (1, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1))
        del buf58
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21.run(buf61, arg31_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg31_1
        buf62 = reinterpret_tensor(buf59, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf59  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg32_1, buf62, 8192, grid=grid(8192), stream=stream0)
        del arg32_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf63 = extern_kernels.convolution(buf61, buf62, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf61
        buf64 = buf49; del buf49  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22.run(buf64, buf63, arg33_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg33_1
        del buf63
        buf65 = buf50; del buf50  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg34_1, buf65, 1728, grid=grid(1728), stream=stream0)
        del arg34_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf66, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        buf67 = buf54; del buf54  # reuse
        buf68 = buf53; del buf53  # reuse
        buf69 = buf52; del buf52  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_17.run(buf66, arg35_1, buf67, buf68, buf69, 320, 52429, grid=grid(320), stream=stream0)
        buf70 = buf56; del buf56  # reuse
        buf71 = buf55; del buf55  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf67, buf68, buf69, buf70, buf71, 64, 5, grid=grid(64), stream=stream0)
        buf73 = buf66; del buf66  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_19.run(buf73, arg35_1, buf70, buf71, arg36_1, arg37_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        buf74 = reinterpret_tensor(buf62, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf62  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg38_1, buf74, 8192, grid=grid(8192), stream=stream0)
        del arg38_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (1, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1))
        del buf73
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21.run(buf76, arg39_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg39_1
        buf77 = reinterpret_tensor(buf74, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf74  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg40_1, buf77, 8192, grid=grid(8192), stream=stream0)
        del arg40_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf76
        buf79 = buf64; del buf64  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22.run(buf79, buf78, arg41_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg41_1
        del buf78
        buf80 = buf65; del buf65  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg42_1, buf80, 1728, grid=grid(1728), stream=stream0)
        del arg42_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf81, (1, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf82 = reinterpret_tensor(buf37, (1, 64, 1, 1, 4), (256, 4, 256, 256, 1), 0); del buf37  # reuse
        buf83 = reinterpret_tensor(buf36, (1, 64, 1, 1, 4), (256, 4, 256, 256, 1), 0); del buf36  # reuse
        buf84 = reinterpret_tensor(buf35, (1, 64, 1, 1, 4), (256, 4, 256, 256, 1), 0); del buf35  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_23.run(buf81, arg43_1, buf82, buf83, buf84, 256, 8192, grid=grid(256), stream=stream0)
        buf85 = buf71; del buf71  # reuse
        buf86 = buf70; del buf70  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf82, buf83, buf84, buf85, buf86, 64, 4, grid=grid(64), stream=stream0)
        del buf82
        del buf83
        del buf84
        buf88 = buf81; del buf81  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_25.run(buf88, arg43_1, buf85, buf86, arg44_1, arg45_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        del arg44_1
        del arg45_1
        buf89 = reinterpret_tensor(buf77, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf77  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg46_1, buf89, 8192, grid=grid(8192), stream=stream0)
        del arg46_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf88
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_26.run(buf91, arg47_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg47_1
        buf92 = empty_strided_cuda((128, 128, 1, 1, 1), (128, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_27.run(arg48_1, buf92, 16384, grid=grid(16384), stream=stream0)
        del arg48_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf91
        buf94 = buf89; del buf89  # reuse
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg50_1, buf94, 8192, grid=grid(8192), stream=stream0)
        del arg50_1
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy, aten.convolution]
        buf95 = extern_kernels.convolution(buf79, buf94, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        buf96 = buf93; del buf93  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_28.run(buf96, arg49_1, buf95, arg51_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg49_1
        del arg51_1
        del buf95
        buf97 = empty_strided_cuda((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), torch.float16)
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg52_1, buf97, 3456, grid=grid(3456), stream=stream0)
        del arg52_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf98, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        buf99 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 4, 512, 512, 1), torch.float32)
        buf100 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 4, 512, 512, 1), torch.float32)
        buf101 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 4, 512, 512, 1), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_30.run(buf98, arg53_1, buf99, buf100, buf101, 512, 8192, grid=grid(512), stream=stream0)
        buf102 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf103 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf99, buf100, buf101, buf102, buf103, 128, 4, grid=grid(128), stream=stream0)
        buf105 = buf98; del buf98  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_32.run(buf105, arg53_1, buf102, buf103, arg54_1, arg55_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg53_1
        del arg54_1
        del arg55_1
        buf106 = empty_strided_cuda((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg56_1, buf106, 32768, grid=grid(32768), stream=stream0)
        del arg56_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf107 = extern_kernels.convolution(buf105, buf106, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (1, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1))
        del buf105
        buf108 = buf107; del buf107  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34.run(buf108, arg57_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg57_1
        buf109 = reinterpret_tensor(buf106, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf106  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg58_1, buf109, 32768, grid=grid(32768), stream=stream0)
        del arg58_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf110 = extern_kernels.convolution(buf108, buf109, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf108
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_35.run(buf111, buf96, arg59_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg59_1
        del buf96
        buf112 = buf97; del buf97  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg60_1, buf112, 3456, grid=grid(3456), stream=stream0)
        del arg60_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf113, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        buf114 = buf99; del buf99  # reuse
        buf115 = buf101; del buf101  # reuse
        buf116 = buf100; del buf100  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_30.run(buf113, arg61_1, buf114, buf115, buf116, 512, 8192, grid=grid(512), stream=stream0)
        buf117 = buf103; del buf103  # reuse
        buf118 = buf102; del buf102  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf114, buf115, buf116, buf117, buf118, 128, 4, grid=grid(128), stream=stream0)
        buf120 = buf113; del buf113  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_32.run(buf120, arg61_1, buf117, buf118, arg62_1, arg63_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf121 = reinterpret_tensor(buf109, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf109  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg64_1, buf121, 32768, grid=grid(32768), stream=stream0)
        del arg64_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (1, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1))
        del buf120
        buf123 = buf122; del buf122  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34.run(buf123, arg65_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg65_1
        buf124 = reinterpret_tensor(buf121, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf121  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg66_1, buf124, 32768, grid=grid(32768), stream=stream0)
        del arg66_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf123
        buf126 = buf111; del buf111  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_36.run(buf126, buf125, arg67_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg67_1
        del buf125
        buf127 = buf112; del buf112  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg68_1, buf127, 3456, grid=grid(3456), stream=stream0)
        del arg68_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf128 = extern_kernels.convolution(buf126, buf127, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf128, (1, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf132 = empty_strided_cuda((1, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float16)
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_37.run(buf128, arg69_1, arg70_1, arg71_1, buf132, 128, 4096, grid=grid(128), stream=stream0)
        del arg69_1
        del arg70_1
        del arg71_1
        del buf128
        buf133 = reinterpret_tensor(buf124, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf124  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg72_1, buf133, 32768, grid=grid(32768), stream=stream0)
        del arg72_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf135 = buf134; del buf134  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_38.run(buf135, arg73_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg73_1
        buf136 = empty_strided_cuda((256, 256, 1, 1, 1), (256, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_39.run(arg74_1, buf136, 65536, grid=grid(65536), stream=stream0)
        del arg74_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        del buf135
        buf138 = buf133; del buf133  # reuse
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg76_1, buf138, 32768, grid=grid(32768), stream=stream0)
        del arg76_1
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy, aten.convolution]
        buf139 = extern_kernels.convolution(buf126, buf138, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf140 = buf137; del buf137  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_40.run(buf140, arg75_1, buf139, arg77_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg75_1
        del arg77_1
        buf141 = empty_strided_cuda((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), torch.float16)
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg78_1, buf141, 6912, grid=grid(6912), stream=stream0)
        del arg78_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf142 = extern_kernels.convolution(buf140, buf141, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf142, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf146 = buf139; del buf139  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_42.run(buf142, arg79_1, arg80_1, arg81_1, buf146, 256, 4096, grid=grid(256), stream=stream0)
        del arg79_1
        del arg80_1
        del arg81_1
        del buf142
        buf147 = empty_strided_cuda((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg82_1, buf147, 131072, grid=grid(131072), stream=stream0)
        del arg82_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf148 = extern_kernels.convolution(buf146, buf147, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (1, 512, 16, 16, 16), (2097152, 4096, 256, 16, 1))
        del buf146
        buf149 = buf148; del buf148  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44.run(buf149, arg83_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg83_1
        buf150 = reinterpret_tensor(buf147, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf147  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg84_1, buf150, 131072, grid=grid(131072), stream=stream0)
        del arg84_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf151 = extern_kernels.convolution(buf149, buf150, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        del buf149
        buf152 = buf140; del buf140  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45.run(buf152, buf151, arg85_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg85_1
        buf153 = buf141; del buf141  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg86_1, buf153, 6912, grid=grid(6912), stream=stream0)
        del arg86_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf154, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf158 = buf151; del buf151  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_42.run(buf154, arg87_1, arg88_1, arg89_1, buf158, 256, 4096, grid=grid(256), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del buf154
        buf159 = reinterpret_tensor(buf150, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf150  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg90_1, buf159, 131072, grid=grid(131072), stream=stream0)
        del arg90_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (1, 512, 16, 16, 16), (2097152, 4096, 256, 16, 1))
        del buf158
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44.run(buf161, arg91_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg91_1
        buf162 = reinterpret_tensor(buf159, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf159  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg92_1, buf162, 131072, grid=grid(131072), stream=stream0)
        del arg92_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf163 = extern_kernels.convolution(buf161, buf162, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        del buf161
        buf164 = buf152; del buf152  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45.run(buf164, buf163, arg93_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg93_1
        buf165 = buf153; del buf153  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg94_1, buf165, 6912, grid=grid(6912), stream=stream0)
        del arg94_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf166 = extern_kernels.convolution(buf164, buf165, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf166, (1, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf170 = reinterpret_tensor(buf162, (1, 256, 8, 8, 8), (131072, 512, 64, 8, 1), 0); del buf162  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_per_fused__to_copy_native_group_norm_46.run(buf166, arg95_1, arg96_1, arg97_1, buf170, 256, 512, grid=grid(256), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        buf171 = reinterpret_tensor(buf166, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf166  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg98_1, buf171, 131072, grid=grid(131072), stream=stream0)
        del arg98_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf172 = extern_kernels.convolution(buf170, buf171, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        del buf170
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_47.run(buf173, arg99_1, 262144, grid=grid(262144), stream=stream0)
        del arg99_1
        buf174 = empty_strided_cuda((512, 512, 1, 1, 1), (512, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_48.run(arg100_1, buf174, 262144, grid=grid(262144), stream=stream0)
        del arg100_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        del buf173
        del buf174
        buf176 = buf171; del buf171  # reuse
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg102_1, buf176, 131072, grid=grid(131072), stream=stream0)
        del arg102_1
        # Source Nodes: [conv3d_3], Original ATen: [aten._to_copy, aten.convolution]
        buf177 = extern_kernels.convolution(buf164, buf176, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        buf178 = buf175; del buf175  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, conv3d_3, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_49.run(buf178, arg101_1, buf177, arg103_1, 262144, grid=grid(262144), stream=stream0)
        del arg101_1
        del arg103_1
        buf179 = empty_strided_cuda((512, 1, 3, 3, 3), (27, 27, 9, 3, 1), torch.float16)
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_50.run(arg104_1, buf179, 13824, grid=grid(13824), stream=stream0)
        del arg104_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf180 = extern_kernels.convolution(buf178, buf179, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=512, bias=None)
        assert_size_stride(buf180, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        buf184 = buf177; del buf177  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_per_fused__to_copy_native_group_norm_51.run(buf180, arg105_1, arg106_1, arg107_1, buf184, 512, 512, grid=grid(512), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del buf180
        buf185 = reinterpret_tensor(buf132, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf132  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_52.run(arg108_1, buf185, 524288, grid=grid(524288), stream=stream0)
        del arg108_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf186 = extern_kernels.convolution(buf184, buf185, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (1, 1024, 8, 8, 8), (524288, 512, 64, 8, 1))
        del buf184
        buf187 = buf186; del buf186  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_53.run(buf187, arg109_1, 524288, grid=grid(524288), stream=stream0)
        del arg109_1
        buf188 = reinterpret_tensor(buf185, (512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), 0); del buf185  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_52.run(arg110_1, buf188, 524288, grid=grid(524288), stream=stream0)
        del arg110_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        del buf187
        buf190 = buf178; del buf178  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_54.run(buf190, buf189, arg111_1, 262144, grid=grid(262144), stream=stream0)
        del arg111_1
        buf191 = buf179; del buf179  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_50.run(arg112_1, buf191, 13824, grid=grid(13824), stream=stream0)
        del arg112_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=512, bias=None)
        assert_size_stride(buf192, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        buf196 = buf189; del buf189  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_per_fused__to_copy_native_group_norm_51.run(buf192, arg113_1, arg114_1, arg115_1, buf196, 512, 512, grid=grid(512), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        del buf192
        buf197 = reinterpret_tensor(buf188, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf188  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_52.run(arg116_1, buf197, 524288, grid=grid(524288), stream=stream0)
        del arg116_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (1, 1024, 8, 8, 8), (524288, 512, 64, 8, 1))
        del buf196
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_53.run(buf199, arg117_1, 524288, grid=grid(524288), stream=stream0)
        del arg117_1
        buf200 = reinterpret_tensor(buf197, (512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), 0); del buf197  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_52.run(arg118_1, buf200, 524288, grid=grid(524288), stream=stream0)
        del arg118_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf201 = extern_kernels.convolution(buf199, buf200, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (1, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        del buf199
        buf202 = buf190; del buf190  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_54.run(buf202, buf201, arg119_1, 262144, grid=grid(262144), stream=stream0)
        del arg119_1
        buf203 = buf191; del buf191  # reuse
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_50.run(arg120_1, buf203, 13824, grid=grid(13824), stream=stream0)
        del arg120_1
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy, aten.convolution]
        buf204 = extern_kernels.convolution(buf202, buf203, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=512, bias=None)
        assert_size_stride(buf204, (1, 512, 15, 15, 15), (1728000, 3375, 225, 15, 1))
        del buf203
        buf208 = empty_strided_cuda((1, 512, 15, 15, 15), (1728000, 3375, 225, 15, 1), torch.float16)
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_55.run(buf204, arg121_1, arg122_1, arg123_1, buf208, 512, 3375, grid=grid(512), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del buf204
        buf209 = reinterpret_tensor(buf200, (1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf200  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_52.run(arg124_1, buf209, 524288, grid=grid(524288), stream=stream0)
        del arg124_1
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (1, 1024, 15, 15, 15), (3456000, 3375, 225, 15, 1))
        del buf208
        del buf209
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_56.run(buf211, arg125_1, 3456000, grid=grid(3456000), stream=stream0)
        del arg125_1
        buf212 = reinterpret_tensor(buf201, (256, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), 0); del buf201  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_48.run(arg126_1, buf212, 262144, grid=grid(262144), stream=stream0)
        del arg126_1
        # Source Nodes: [conv3d, conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (1, 256, 15, 15, 15), (864000, 3375, 225, 15, 1))
        del buf211
        del buf212
        buf214 = buf176; del buf176  # reuse
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg128_1, buf214, 131072, grid=grid(131072), stream=stream0)
        del arg128_1
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy, aten.convolution]
        buf215 = extern_kernels.convolution(buf202, buf214, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (1, 256, 15, 15, 15), (864000, 3375, 225, 15, 1))
        del buf202
        buf216 = buf164; del buf164  # reuse
        # Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_57.run(buf216, buf213, arg127_1, buf215, arg129_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg127_1
        del arg129_1
        del buf213
        del buf215
        buf217 = buf165; del buf165  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg130_1, buf217, 6912, grid=grid(6912), stream=stream0)
        del arg130_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf218 = extern_kernels.convolution(buf216, buf217, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf218, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf222 = buf163; del buf163  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_42.run(buf218, arg131_1, arg132_1, arg133_1, buf222, 256, 4096, grid=grid(256), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del buf218
        buf223 = buf214; del buf214  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg134_1, buf223, 131072, grid=grid(131072), stream=stream0)
        del arg134_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (1, 512, 16, 16, 16), (2097152, 4096, 256, 16, 1))
        del buf222
        buf225 = buf224; del buf224  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44.run(buf225, arg135_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg135_1
        buf226 = reinterpret_tensor(buf223, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf223  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg136_1, buf226, 131072, grid=grid(131072), stream=stream0)
        del arg136_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf227 = extern_kernels.convolution(buf225, buf226, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        del buf225
        buf228 = buf216; del buf216  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45.run(buf228, buf227, arg137_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg137_1
        buf229 = buf217; del buf217  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg138_1, buf229, 6912, grid=grid(6912), stream=stream0)
        del arg138_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf230 = extern_kernels.convolution(buf228, buf229, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf230, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf234 = buf227; del buf227  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_42.run(buf230, arg139_1, arg140_1, arg141_1, buf234, 256, 4096, grid=grid(256), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del buf230
        buf235 = reinterpret_tensor(buf226, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf226  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg142_1, buf235, 131072, grid=grid(131072), stream=stream0)
        del arg142_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf236 = extern_kernels.convolution(buf234, buf235, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (1, 512, 16, 16, 16), (2097152, 4096, 256, 16, 1))
        del buf234
        buf237 = buf236; del buf236  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_44.run(buf237, arg143_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg143_1
        buf238 = reinterpret_tensor(buf235, (256, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf235  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg144_1, buf238, 131072, grid=grid(131072), stream=stream0)
        del arg144_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf239 = extern_kernels.convolution(buf237, buf238, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (1, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        del buf237
        buf240 = buf228; del buf228  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_45.run(buf240, buf239, arg145_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg145_1
        del buf239
        buf241 = buf229; del buf229  # reuse
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_41.run(arg146_1, buf241, 6912, grid=grid(6912), stream=stream0)
        del arg146_1
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy, aten.convolution]
        buf242 = extern_kernels.convolution(buf240, buf241, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf242, (1, 256, 31, 31, 31), (7626496, 29791, 961, 31, 1))
        del buf241
        buf246 = empty_strided_cuda((1, 256, 31, 31, 31), (7626496, 29791, 961, 31, 1), torch.float16)
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_red_fused__to_copy_native_group_norm_58.run(buf242, arg147_1, arg148_1, arg149_1, buf246, 256, 29791, grid=grid(256), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del buf242
        buf247 = reinterpret_tensor(buf238, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf238  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_43.run(arg150_1, buf247, 131072, grid=grid(131072), stream=stream0)
        del arg150_1
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf248 = extern_kernels.convolution(buf246, buf247, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (1, 512, 31, 31, 31), (15252992, 29791, 961, 31, 1))
        del buf246
        del buf247
        buf249 = buf248; del buf248  # reuse
        # Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_59.run(buf249, arg151_1, 15252992, grid=grid(15252992), stream=stream0)
        del arg151_1
        buf250 = reinterpret_tensor(buf136, (128, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0); del buf136  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_39.run(arg152_1, buf250, 65536, grid=grid(65536), stream=stream0)
        del arg152_1
        # Source Nodes: [conv3d, conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (1, 128, 31, 31, 31), (3813248, 29791, 961, 31, 1))
        del buf249
        del buf250
        buf252 = buf138; del buf138  # reuse
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg154_1, buf252, 32768, grid=grid(32768), stream=stream0)
        del arg154_1
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy, aten.convolution]
        buf253 = extern_kernels.convolution(buf240, buf252, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (1, 128, 31, 31, 31), (3813248, 29791, 961, 31, 1))
        del buf240
        buf254 = buf126; del buf126  # reuse
        # Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_1, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_60.run(buf254, buf251, arg153_1, buf253, arg155_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg153_1
        del arg155_1
        del buf251
        del buf253
        buf255 = buf127; del buf127  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg156_1, buf255, 3456, grid=grid(3456), stream=stream0)
        del arg156_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf256 = extern_kernels.convolution(buf254, buf255, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf256, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        buf257 = buf116; del buf116  # reuse
        buf258 = buf115; del buf115  # reuse
        buf259 = buf114; del buf114  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_30.run(buf256, arg157_1, buf257, buf258, buf259, 512, 8192, grid=grid(512), stream=stream0)
        buf260 = buf118; del buf118  # reuse
        buf261 = buf117; del buf117  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf257, buf258, buf259, buf260, buf261, 128, 4, grid=grid(128), stream=stream0)
        buf263 = buf256; del buf256  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_32.run(buf263, arg157_1, buf260, buf261, arg158_1, arg159_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        buf264 = buf252; del buf252  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg160_1, buf264, 32768, grid=grid(32768), stream=stream0)
        del arg160_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf265 = extern_kernels.convolution(buf263, buf264, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (1, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1))
        del buf263
        buf266 = buf265; del buf265  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34.run(buf266, arg161_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg161_1
        buf267 = reinterpret_tensor(buf264, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf264  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg162_1, buf267, 32768, grid=grid(32768), stream=stream0)
        del arg162_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf266
        buf269 = buf254; del buf254  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_36.run(buf269, buf268, arg163_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg163_1
        del buf268
        buf270 = buf255; del buf255  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg164_1, buf270, 3456, grid=grid(3456), stream=stream0)
        del arg164_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf271 = extern_kernels.convolution(buf269, buf270, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf271, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        buf272 = buf259; del buf259  # reuse
        buf273 = buf258; del buf258  # reuse
        buf274 = buf257; del buf257  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_30.run(buf271, arg165_1, buf272, buf273, buf274, 512, 8192, grid=grid(512), stream=stream0)
        buf275 = buf261; del buf261  # reuse
        buf276 = buf260; del buf260  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf272, buf273, buf274, buf275, buf276, 128, 4, grid=grid(128), stream=stream0)
        del buf272
        del buf273
        del buf274
        buf278 = buf271; del buf271  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_32.run(buf278, arg165_1, buf275, buf276, arg166_1, arg167_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg165_1
        del arg166_1
        del arg167_1
        buf279 = reinterpret_tensor(buf267, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf267  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg168_1, buf279, 32768, grid=grid(32768), stream=stream0)
        del arg168_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf280 = extern_kernels.convolution(buf278, buf279, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (1, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1))
        del buf278
        buf281 = buf280; del buf280  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_34.run(buf281, arg169_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg169_1
        buf282 = reinterpret_tensor(buf279, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf279  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg170_1, buf282, 32768, grid=grid(32768), stream=stream0)
        del arg170_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf283 = extern_kernels.convolution(buf281, buf282, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (1, 128, 32, 32, 32), (4194304, 32768, 1024, 32, 1))
        del buf281
        buf284 = buf269; del buf269  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_36.run(buf284, buf283, arg171_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg171_1
        del buf283
        buf285 = buf270; del buf270  # reuse
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(arg172_1, buf285, 3456, grid=grid(3456), stream=stream0)
        del arg172_1
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy, aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf286, (1, 128, 63, 63, 63), (32006016, 250047, 3969, 63, 1))
        del buf285
        buf287 = empty_strided_cuda((1, 128, 1, 1, 3), (384, 3, 384, 384, 1), torch.float32)
        buf288 = empty_strided_cuda((1, 128, 1, 1, 3), (384, 3, 384, 384, 1), torch.float32)
        buf289 = empty_strided_cuda((1, 128, 1, 1, 3), (384, 3, 384, 384, 1), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_61.run(buf286, arg173_1, buf287, buf288, buf289, 384, 83349, grid=grid(384), stream=stream0)
        buf290 = buf276; del buf276  # reuse
        buf291 = buf275; del buf275  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_62.run(buf287, buf288, buf289, buf290, buf291, 128, 3, grid=grid(128), stream=stream0)
        del buf287
        del buf288
        del buf289
        buf293 = buf286; del buf286  # reuse
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_63.run(buf293, arg173_1, buf290, buf291, arg174_1, arg175_1, 32006016, grid=grid(32006016), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        del buf290
        del buf291
        buf294 = reinterpret_tensor(buf282, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf282  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_33.run(arg176_1, buf294, 32768, grid=grid(32768), stream=stream0)
        del arg176_1
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf295 = extern_kernels.convolution(buf293, buf294, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (1, 256, 63, 63, 63), (64012032, 250047, 3969, 63, 1))
        del buf293
        del buf294
        buf296 = buf295; del buf295  # reuse
        # Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_64.run(buf296, arg177_1, 64012032, grid=grid(64012032), stream=stream0)
        del arg177_1
        buf297 = reinterpret_tensor(buf92, (64, 256, 1, 1, 1), (256, 1, 1, 1, 1), 0); del buf92  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_27.run(arg178_1, buf297, 16384, grid=grid(16384), stream=stream0)
        del arg178_1
        # Source Nodes: [conv3d, conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf298 = extern_kernels.convolution(buf296, buf297, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (1, 64, 63, 63, 63), (16003008, 250047, 3969, 63, 1))
        del buf296
        del buf297
        buf299 = buf94; del buf94  # reuse
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg180_1, buf299, 8192, grid=grid(8192), stream=stream0)
        del arg180_1
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy, aten.convolution]
        buf300 = extern_kernels.convolution(buf284, buf299, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (1, 64, 63, 63, 63), (16003008, 250047, 3969, 63, 1))
        del buf284
        buf301 = buf79; del buf79  # reuse
        # Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_2, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_65.run(buf301, buf298, arg179_1, buf300, arg181_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg179_1
        del arg181_1
        del buf298
        del buf300
        buf302 = buf80; del buf80  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg182_1, buf302, 1728, grid=grid(1728), stream=stream0)
        del arg182_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf303 = extern_kernels.convolution(buf301, buf302, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf303, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        buf304 = buf69; del buf69  # reuse
        buf305 = buf68; del buf68  # reuse
        buf306 = buf67; del buf67  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_17.run(buf303, arg183_1, buf304, buf305, buf306, 320, 52429, grid=grid(320), stream=stream0)
        buf307 = buf86; del buf86  # reuse
        buf308 = buf85; del buf85  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf304, buf305, buf306, buf307, buf308, 64, 5, grid=grid(64), stream=stream0)
        buf310 = buf303; del buf303  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_19.run(buf310, arg183_1, buf307, buf308, arg184_1, arg185_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        buf311 = buf299; del buf299  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg186_1, buf311, 8192, grid=grid(8192), stream=stream0)
        del arg186_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf312 = extern_kernels.convolution(buf310, buf311, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (1, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1))
        del buf310
        buf313 = buf312; del buf312  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21.run(buf313, arg187_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg187_1
        buf314 = reinterpret_tensor(buf311, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf311  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg188_1, buf314, 8192, grid=grid(8192), stream=stream0)
        del arg188_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf315 = extern_kernels.convolution(buf313, buf314, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf313
        buf316 = buf301; del buf301  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22.run(buf316, buf315, arg189_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg189_1
        del buf315
        buf317 = buf302; del buf302  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg190_1, buf317, 1728, grid=grid(1728), stream=stream0)
        del arg190_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf318, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        buf319 = buf306; del buf306  # reuse
        buf320 = buf305; del buf305  # reuse
        buf321 = buf304; del buf304  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_17.run(buf318, arg191_1, buf319, buf320, buf321, 320, 52429, grid=grid(320), stream=stream0)
        buf322 = buf308; del buf308  # reuse
        buf323 = buf307; del buf307  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_18.run(buf319, buf320, buf321, buf322, buf323, 64, 5, grid=grid(64), stream=stream0)
        buf325 = buf318; del buf318  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_19.run(buf325, arg191_1, buf322, buf323, arg192_1, arg193_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg191_1
        del arg192_1
        del arg193_1
        buf326 = reinterpret_tensor(buf314, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf314  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg194_1, buf326, 8192, grid=grid(8192), stream=stream0)
        del arg194_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf327 = extern_kernels.convolution(buf325, buf326, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (1, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1))
        del buf325
        buf328 = buf327; del buf327  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_21.run(buf328, arg195_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg195_1
        buf329 = reinterpret_tensor(buf326, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf326  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg196_1, buf329, 8192, grid=grid(8192), stream=stream0)
        del arg196_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf330 = extern_kernels.convolution(buf328, buf329, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (1, 64, 64, 64, 64), (16777216, 262144, 4096, 64, 1))
        del buf328
        buf331 = buf316; del buf316  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_22.run(buf331, buf330, arg197_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg197_1
        del buf330
        buf332 = buf317; del buf317  # reuse
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16.run(arg198_1, buf332, 1728, grid=grid(1728), stream=stream0)
        del arg198_1
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy, aten.convolution]
        buf333 = extern_kernels.convolution(buf331, buf332, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf333, (1, 64, 127, 127, 127), (131096512, 2048383, 16129, 127, 1))
        del buf332
        buf334 = empty_strided_cuda((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), torch.float32)
        buf335 = empty_strided_cuda((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), torch.float32)
        buf336 = empty_strided_cuda((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_66.run(buf333, arg199_1, buf334, buf335, buf336, 1024, 128024, grid=grid(1024), stream=stream0)
        buf337 = buf323; del buf323  # reuse
        buf338 = buf322; del buf322  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_67.run(buf334, buf335, buf336, buf337, buf338, 64, 16, grid=grid(64), stream=stream0)
        del buf334
        del buf335
        del buf336
        buf340 = buf333; del buf333  # reuse
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_68.run(buf340, arg199_1, buf337, buf338, arg200_1, arg201_1, 131096512, grid=grid(131096512), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        del buf337
        del buf338
        buf341 = reinterpret_tensor(buf329, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf329  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_20.run(arg202_1, buf341, 8192, grid=grid(8192), stream=stream0)
        del arg202_1
        # Source Nodes: [conv3d, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (1, 128, 127, 127, 127), (262193024, 2048383, 16129, 127, 1))
        del buf340
        del buf341
        buf343 = buf342; del buf342  # reuse
        # Source Nodes: [conv3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_69.run(buf343, arg203_1, 262193024, grid=grid(262193024), stream=stream0)
        del arg203_1
        buf344 = reinterpret_tensor(buf45, (32, 128, 1, 1, 1), (128, 1, 1, 1, 1), 0); del buf45  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_14.run(arg204_1, buf344, 4096, grid=grid(4096), stream=stream0)
        del arg204_1
        # Source Nodes: [conv3d, conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf345 = extern_kernels.convolution(buf343, buf344, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (1, 32, 127, 127, 127), (65548256, 2048383, 16129, 127, 1))
        del buf343
        del buf344
        buf346 = buf47; del buf47  # reuse
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg206_1, buf346, 2048, grid=grid(2048), stream=stream0)
        del arg206_1
        # Source Nodes: [conv_transpose3d_1], Original ATen: [aten._to_copy, aten.convolution]
        buf347 = extern_kernels.convolution(buf331, buf346, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (1, 32, 127, 127, 127), (65548256, 2048383, 16129, 127, 1))
        del buf331
        buf348 = buf32; del buf32  # reuse
        # Source Nodes: [add, conv3d, conv3d_1, conv_transpose3d_1, dec_x_3, gelu, group_norm, pad, pad_1], Original ATen: [aten._to_copy, aten.add, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_constant_pad_nd_convolution_gelu_native_group_norm_70.run(buf348, buf345, arg205_1, buf347, arg207_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg205_1
        del arg207_1
        del buf345
        del buf347
        buf349 = buf33; del buf33  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(arg208_1, buf349, 864, grid=grid(864), stream=stream0)
        del arg208_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf350 = extern_kernels.convolution(buf348, buf349, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf350, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        buf351 = reinterpret_tensor(buf321, (1, 32, 1, 1, 10), (320, 10, 320, 320, 1), 0); del buf321  # reuse
        buf352 = reinterpret_tensor(buf320, (1, 32, 1, 1, 10), (320, 10, 320, 320, 1), 0); del buf320  # reuse
        buf353 = reinterpret_tensor(buf319, (1, 32, 1, 1, 10), (320, 10, 320, 320, 1), 0); del buf319  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_3.run(buf350, arg209_1, buf351, buf352, buf353, 320, 209716, grid=grid(320), stream=stream0)
        buf354 = buf39; del buf39  # reuse
        buf355 = buf38; del buf38  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf351, buf352, buf353, buf354, buf355, 32, 10, grid=grid(32), stream=stream0)
        buf357 = buf350; del buf350  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_5.run(buf357, arg209_1, buf354, buf355, arg210_1, arg211_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg209_1
        del arg210_1
        del arg211_1
        buf358 = buf346; del buf346  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg212_1, buf358, 2048, grid=grid(2048), stream=stream0)
        del arg212_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf359 = extern_kernels.convolution(buf357, buf358, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (1, 64, 128, 128, 128), (134217728, 2097152, 16384, 128, 1))
        del buf357
        buf360 = buf359; del buf359  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7.run(buf360, arg213_1, 134217728, grid=grid(134217728), stream=stream0)
        del arg213_1
        buf361 = reinterpret_tensor(buf358, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf358  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg214_1, buf361, 2048, grid=grid(2048), stream=stream0)
        del arg214_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf362 = extern_kernels.convolution(buf360, buf361, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        del buf360
        buf363 = buf348; del buf348  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_9.run(buf363, buf362, arg215_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg215_1
        del buf362
        buf364 = buf349; del buf349  # reuse
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(arg216_1, buf364, 864, grid=grid(864), stream=stream0)
        del arg216_1
        # Source Nodes: [conv3d], Original ATen: [aten._to_copy, aten.convolution]
        buf365 = extern_kernels.convolution(buf363, buf364, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf365, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        del buf364
        buf366 = buf353; del buf353  # reuse
        buf367 = buf352; del buf352  # reuse
        buf368 = buf351; del buf351  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_3.run(buf365, arg217_1, buf366, buf367, buf368, 320, 209716, grid=grid(320), stream=stream0)
        buf369 = buf355; del buf355  # reuse
        buf370 = buf354; del buf354  # reuse
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf366, buf367, buf368, buf369, buf370, 32, 10, grid=grid(32), stream=stream0)
        del buf366
        del buf367
        del buf368
        buf372 = buf365; del buf365  # reuse
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.native_group_norm]
        triton_poi_fused__to_copy_native_group_norm_5.run(buf372, arg217_1, buf369, buf370, arg218_1, arg219_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del buf369
        del buf370
        buf373 = reinterpret_tensor(buf361, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1), 0); del buf361  # reuse
        # Source Nodes: [conv3d_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg220_1, buf373, 2048, grid=grid(2048), stream=stream0)
        del arg220_1
        # Source Nodes: [conv3d_1, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.native_group_norm]
        buf374 = extern_kernels.convolution(buf372, buf373, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (1, 64, 128, 128, 128), (134217728, 2097152, 16384, 128, 1))
        del buf372
        buf375 = buf374; del buf374  # reuse
        # Source Nodes: [conv3d_1, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_convolution_gelu_native_group_norm_7.run(buf375, arg221_1, 134217728, grid=grid(134217728), stream=stream0)
        del arg221_1
        buf376 = reinterpret_tensor(buf373, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1), 0); del buf373  # reuse
        # Source Nodes: [conv3d_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6.run(arg222_1, buf376, 2048, grid=grid(2048), stream=stream0)
        del arg222_1
        # Source Nodes: [conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.native_group_norm]
        buf377 = extern_kernels.convolution(buf375, buf376, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (1, 32, 128, 128, 128), (67108864, 2097152, 16384, 128, 1))
        del buf375
        del buf376
        buf378 = buf363; del buf363  # reuse
        # Source Nodes: [add, conv3d_1, conv3d_2, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_9.run(buf378, buf377, arg223_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg223_1
        del buf377
        buf379 = empty_strided_cuda((32, 7, 1, 1, 1), (7, 1, 1, 1, 1), torch.float16)
        # Source Nodes: [conv_transpose3d], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_71.run(arg224_1, buf379, 224, grid=grid(224), stream=stream0)
        del arg224_1
        # Source Nodes: [add, conv3d_1, conv3d_2, conv_transpose3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        buf380 = extern_kernels.convolution(buf378, buf379, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (1, 7, 128, 128, 128), (14680064, 2097152, 16384, 128, 1))
        del buf378
        del buf379
        buf381 = empty_strided_cuda((1, 7, 128, 128, 128), (14680064, 1, 114688, 896, 7), torch.float16)
        # Source Nodes: [add, conv3d_1, conv3d_2, conv_transpose3d, gelu, group_norm], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused__to_copy_add_convolution_gelu_native_group_norm_72.run(buf380, arg225_1, buf381, 7, 2097152, grid=grid(7, 2097152), stream=stream0)
        del arg225_1
        del buf380
    return (buf381, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 1024, 1, 1, 1), (1024, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, 512, 1, 1, 1), (512, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((64, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((32, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((32, 7, 1, 1, 1), (7, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((32, 1, 1, 1, 1), (1, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1, 1, 128, 128, 128), (2097152, 1, 16384, 128, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
