from hisim.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H100_SXM = AcceleratorInfo.from_dict(
        config={
            "name": "h100_sxm",
            "device_alias": ["H100_SXM", "h100"],
            "tflops": {
                "FP8_TENSOR": 4667,
                "INT8_TENSOR": 4667,
                "FP16_TENSOR": 2333,
                "BF16_TENSOR": 2333,
                "FP32": 1167,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 3350,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h100/",
        },
        save_to_registry=True,
    )

    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "h20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )

    NVIDIA_RTX_PRO_6000_SERVER = AcceleratorInfo.from_dict(
        config={
            "name": "rtx_pro_6000_server",
            "device_alias": [
                "RTX_PRO_6000_SERVER",
                "rtx6000_server",
            ],
            "tflops": {
                "FP8_TENSOR": 935.6,
                "INT8_TENSOR": 935.6,
                "FP16_TENSOR": 467.8,
                "BF16_TENSOR": 467.8,
                "FP32": 233.9,
            },
            "hbm_capacity_gb": 102,
            "hbm_bandwidth_gb": 1433,
            "inter_node_bandwidth_gb": 50,
            "intra_node_bandwidth_gb": 64,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/products/workstations/rtx-pro-6000/",
        },
        save_to_registry=True,
    )
