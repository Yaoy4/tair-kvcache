from hisim.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H100_SXM = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H100 SXM",
            "device_alias": ["H100", "h100_sxm"],
            "tflops": {
                "FP8_TENSOR": 1978,
                "INT8_TENSOR": 1978,
                "FP16_TENSOR": 989,
                "BF16_TENSOR": 989,
                "FP32": 67,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 3350,
            "inter_node_bandwidth_gb": 25,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h100/",
        },
        save_to_registry=True,
    )

    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
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

    NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "device_alias": [
                "RTX 6000 PRO",
                "rtx 6000 pro",
                "RTX PRO 6000",
                "rtx pro 6000",
                "rtx_pro_6000_server",
            ],
            "tflops": {
                "BF16_TENSOR": 467.800534963698,
                "INT8_TENSOR": 935.601069927397,
                "FP8_TENSOR": 935.601069927397,
                "FP4_TENSOR": 1871.202139854795,
            },
            "hbm_capacity_gb": 102.641958912,
            "hbm_bandwidth_gb": 1792,
            "inter_node_bandwidth_gb": 50,
            "intra_node_bandwidth_gb": 64,
            "vendor": "NVIDIA",
        },
        save_to_registry=True,
    )


class Intel:
    INTEL_B60 = AcceleratorInfo.from_dict(
        config={
            "name": "Intel B60",
            "device_alias": ["B60", "b60", "Intel B60", "intel_b60"],
            "tflops": {
                "BF16_TENSOR": 98.3,
                "INT8_TENSOR": 197,
                "FP32": 12.28,
            },
            "hbm_capacity_gb": 24,
            "hbm_bandwidth_gb": 456,
            "inter_node_bandwidth_gb": 50,
            "intra_node_bandwidth_gb": 32,
            "vendor": "Intel",
            "ref": "https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/workstations/b-series.html",
        },
        save_to_registry=True,
    )
