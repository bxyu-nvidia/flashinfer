/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "trtllm_moe_routing_workspace.h"

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_run_routing, flashinfer::trtllm_moe_run_routing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_prepare_routing, flashinfer::trtllm_moe_prepare_routing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_run_finalize, flashinfer::trtllm_moe_run_finalize);
