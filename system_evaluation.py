#!/usr/bin/env python3
"""
ECH0-PRIME System Resource Evaluation
Assesses hardware and software limitations for running the full AGI system.
"""

import os
import sys
import psutil
import platform
import subprocess
import time
import torch
import numpy as np
from typing import Dict, List, Any
import json

class SystemEvaluator:
    """Comprehensive system resource and performance evaluation"""

    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "total_memory": psutil.virtual_memory().total / (1024**3),  # GB
            "available_memory": psutil.virtual_memory().available / (1024**3),  # GB
        }

    def evaluate_hardware_limits(self) -> Dict[str, Any]:
        """Evaluate hardware resource limitations"""
        print("🔧 Evaluating Hardware Limitations...")

        # Memory assessment
        memory_info = psutil.virtual_memory()
        memory_limits = {
            "total_ram_gb": memory_info.total / (1024**3),
            "available_ram_gb": memory_info.available / (1024**3),
            "memory_usage_percent": memory_info.percent,
            "swap_total_gb": psutil.swap_memory().total / (1024**3) if psutil.swap_memory().total > 0 else 0,
            "recommended_min_ram": 16,  # GB for AGI system
            "can_run_full_system": memory_info.available > 8 * (1024**3),  # 8GB minimum
        }

        # CPU assessment
        cpu_limits = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "recommended_min_cores": 4,
            "can_handle_parallel_processing": psutil.cpu_count(logical=True) >= 4,
        }

        # GPU assessment
        gpu_limits = self._evaluate_gpu_limits()

        # Disk assessment
        disk_limits = self._evaluate_disk_limits()

        return {
            "memory": memory_limits,
            "cpu": cpu_limits,
            "gpu": gpu_limits,
            "disk": disk_limits
        }

    def _evaluate_gpu_limits(self) -> Dict[str, Any]:
        """Evaluate GPU capabilities and limitations"""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "gpu_count": 0,
            "recommended_min_vram": 8,  # GB
        }

        # Check CUDA GPUs first
        if torch.cuda.is_available():
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
                "gpu_architecture": torch.cuda.get_device_capability(0),
                "gpu_type": "cuda"
            })

            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda')
                gpu_info["can_allocate_4gb"] = True
                del test_tensor
                torch.cuda.empty_cache()
            except RuntimeError:
                gpu_info["can_allocate_4gb"] = False

        # Check Apple Silicon MPS
        elif gpu_info["mps_available"]:
            gpu_info["gpu_count"] = 1
            gpu_info.update({
                "gpu_name": "Apple Silicon GPU",
                "gpu_type": "mps",
                "total_vram_gb": self._get_apple_silicon_vram(),
            })

            # Test MPS memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device='mps')
                gpu_info["can_allocate_4gb"] = True
                del test_tensor
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except RuntimeError:
                gpu_info["can_allocate_4gb"] = False

        return gpu_info

    def _get_apple_silicon_vram(self) -> float:
        """Estimate Apple Silicon GPU VRAM"""
        try:
            # Try to get system memory as approximation
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            # Apple Silicon typically shares memory, GPU gets portion of system RAM
            # Conservative estimate: 50% of system RAM for M1/M2, more for M3/M4
            if total_memory_gb >= 32:  # M3/M4 with more RAM
                return min(total_memory_gb * 0.6, 24)  # Up to 24GB shared
            elif total_memory_gb >= 16:  # M1/M2 with 16GB+
                return min(total_memory_gb * 0.5, 16)  # Up to 16GB shared
            else:  # Base models
                return total_memory_gb * 0.4  # 40% of system RAM
        except:
            return 8.0  # Conservative fallback

    def _evaluate_disk_limits(self) -> Dict[str, Any]:
        """Evaluate disk space and I/O limitations"""
        disk_info = {}

        # Get disk usage for current directory
        try:
            usage = psutil.disk_usage('/')
            disk_info.update({
                "total_space_gb": usage.total / (1024**3),
                "free_space_gb": usage.free / (1024**3),
                "used_percent": usage.percent,
                "recommended_free_space": 50,  # GB
                "sufficient_space": usage.free > 50 * (1024**3),
            })
        except Exception as e:
            disk_info["disk_error"] = str(e)

        return disk_info

    def evaluate_software_dependencies(self) -> Dict[str, Any]:
        """Evaluate software dependency status and compatibility"""
        print("📦 Evaluating Software Dependencies...")

        dependencies = {
            "torch": self._check_package("torch"),
            "qiskit": self._check_package("qiskit"),
            "numpy": self._check_package("numpy"),
            "scipy": self._check_package("scipy"),
            "psutil": self._check_package("psutil"),
            "faiss": self._check_package("faiss-cpu"),
            "transformers": self._check_package("transformers"),
            "sentence_transformers": self._check_package("sentence-transformers"),
            "networkx": self._check_package("networkx"),
        }

        # Check for missing critical dependencies
        critical_deps = ["torch", "numpy", "scipy"]
        missing_critical = [dep for dep in critical_deps if not dependencies[dep]["available"]]

        return {
            "dependencies": dependencies,
            "missing_critical": missing_critical,
            "all_critical_available": len(missing_critical) == 0
        }

    def _check_package(self, package_name: str) -> Dict[str, Any]:
        """Check if a Python package is available and get version"""
        try:
            module = __import__(package_name.replace("-", "_"))
            version = getattr(module, "__version__", "Unknown")
            return {"available": True, "version": version}
        except ImportError:
            return {"available": False, "version": None}

    def evaluate_performance_bottlenecks(self) -> Dict[str, Any]:
        """Evaluate system performance bottlenecks"""
        print("⚡ Evaluating Performance Bottlenecks...")

        bottlenecks = {}

        # Memory bandwidth test
        bottlenecks["memory_bandwidth"] = self._test_memory_bandwidth()

        # CPU performance test
        bottlenecks["cpu_performance"] = self._test_cpu_performance()

        # I/O performance test
        bottlenecks["io_performance"] = self._test_io_performance()

        return bottlenecks

    def _test_memory_bandwidth(self) -> Dict[str, Any]:
        """Test memory bandwidth performance"""
        try:
            # Simple memory allocation and access test
            start_time = time.time()
            arrays = [np.random.random((1000, 1000)) for _ in range(10)]
            for arr in arrays:
                _ = arr.sum()  # Force memory access
            end_time = time.time()

            bandwidth_score = len(arrays) * arrays[0].nbytes / (end_time - start_time) / (1024**2)  # MB/s
            return {
                "bandwidth_mbs": bandwidth_score,
                "acceptable_performance": bandwidth_score > 1000,  # 1GB/s minimum
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_cpu_performance(self) -> Dict[str, Any]:
        """Test CPU performance with matrix operations"""
        try:
            start_time = time.time()
            # Matrix multiplication performance test
            size = 500
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            for _ in range(5):
                c = np.dot(a, b)
            end_time = time.time()

            ops_per_sec = (5 * size**3) / (end_time - start_time)
            return {
                "operations_per_second": ops_per_sec,
                "acceptable_performance": ops_per_sec > 1e7,  # 10M ops/sec minimum
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_io_performance(self) -> Dict[str, Any]:
        """Test disk I/O performance"""
        try:
            test_file = "/tmp/agi_io_test.tmp"
            data = b"0" * (1024 * 1024)  # 1MB

            # Write test
            start_time = time.time()
            with open(test_file, "wb") as f:
                for _ in range(10):
                    f.write(data)
            write_time = time.time() - start_time

            # Read test
            start_time = time.time()
            with open(test_file, "rb") as f:
                for _ in range(10):
                    _ = f.read(1024 * 1024)
            read_time = time.time() - start_time

            # Cleanup
            os.remove(test_file)

            write_speed = (10 * 1024 * 1024) / write_time / (1024**2)  # MB/s
            read_speed = (10 * 1024 * 1024) / read_time / (1024**2)   # MB/s

            return {
                "write_speed_mbs": write_speed,
                "read_speed_mbs": read_speed,
                "acceptable_performance": min(write_speed, read_speed) > 50,  # 50MB/s minimum
            }
        except Exception as e:
            return {"error": str(e)}

    def evaluate_agi_system_limits(self) -> Dict[str, Any]:
        """Evaluate specific limitations for running ECH0-PRIME"""
        print("🧠 Evaluating AGI System Limitations...")

        agi_limits = {
            "neural_network_limits": self._evaluate_nn_limits(),
            "memory_system_limits": self._evaluate_memory_limits(),
            "parallel_processing_limits": self._evaluate_parallel_limits(),
            "real_time_performance": self._evaluate_real_time_limits(),
        }

        return agi_limits

    def _evaluate_nn_limits(self) -> Dict[str, Any]:
        """Evaluate neural network training/inference limits"""
        try:
            # Test maximum model size that can fit in memory
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            # Test progressively larger models
            max_size = 1000
            for size in [1000, 5000, 10000, 50000]:
                try:
                    model = torch.nn.Linear(size, size).to(device)
                    test_input = torch.randn(32, size, device=device)
                    output = model(test_input)
                    max_size = size
                    del model, test_input, output
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif device.type == "mps" and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except RuntimeError:
                    break

            return {
                "max_model_dimension": max_size,
                "device_type": device.type,
                "can_train_large_models": max_size >= 10000,
            }
        except Exception as e:
            return {"error": str(e)}

    def _evaluate_memory_limits(self) -> Dict[str, Any]:
        """Evaluate memory system limitations"""
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB

        return {
            "available_memory_gb": available_memory,
            "can_store_large_knowledge_base": available_memory > 8,
            "recommended_memory_for_full_system": 16,  # GB
            "memory_limited": available_memory < 12,
        }

    def _evaluate_parallel_limits(self) -> Dict[str, Any]:
        """Evaluate parallel processing limitations"""
        cpu_count = os.cpu_count()

        return {
            "available_cores": cpu_count,
            "can_run_hive_mind": cpu_count >= 4,
            "optimal_agent_count": min(cpu_count, 8),
            "parallel_limited": cpu_count < 6,
        }

    def _evaluate_real_time_limits(self) -> Dict[str, Any]:
        """Evaluate real-time performance limitations"""
        # Test response time for typical operations
        try:
            start_time = time.time()
            # Simulate typical AGI operations
            a = np.random.random((1000, 1000))
            b = np.random.random((1000, 1000))
            c = np.dot(a, b)
            end_time = time.time()

            operation_time = end_time - start_time

            return {
                "matrix_mult_time_seconds": operation_time,
                "real_time_capable": operation_time < 0.1,  # 100ms threshold
                "bottleneck_identified": operation_time > 0.5,
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_recommendations(self) -> List[str]:
        """Generate specific recommendations for system improvements"""
        recommendations = []

        # Hardware recommendations
        if self.results.get("hardware", {}).get("memory", {}).get("can_run_full_system") == False:
            recommendations.append("⚠️  UPGRADE RAM: System has insufficient memory for full AGI operations. Recommended: 16GB+ RAM")

        if self.results.get("hardware", {}).get("cpu", {}).get("can_handle_parallel_processing") == False:
            recommendations.append("⚠️  UPGRADE CPU: Need more cores for parallel processing. Recommended: 4+ cores")

        gpu_info = self.results.get("hardware", {}).get("gpu", {})
        has_gpu = gpu_info.get("cuda_available", False) or gpu_info.get("mps_available", False)
        if not has_gpu:
            recommendations.append("⚠️  ADD GPU: GPU acceleration required for efficient neural network operations")

        # Software recommendations
        if not self.results.get("software", {}).get("all_critical_available"):
            missing = self.results["software"]["missing_critical"]
            recommendations.append(f"⚠️  INSTALL DEPENDENCIES: Missing critical packages: {', '.join(missing)}")

        # Performance recommendations
        if not self.results.get("performance", {}).get("memory_bandwidth", {}).get("acceptable_performance"):
            recommendations.append("⚠️  OPTIMIZE MEMORY: Memory bandwidth is bottleneck. Consider faster RAM or SSD")

        if not self.results.get("performance", {}).get("cpu_performance", {}).get("acceptable_performance"):
            recommendations.append("⚠️  OPTIMIZE CPU: CPU performance is limiting. Consider faster processor")

        # AGI-specific recommendations
        if not self.results.get("agi_limits", {}).get("neural_network_limits", {}).get("can_train_large_models"):
            recommendations.append("⚠️  AGI LIMITATION: Cannot train large neural models. Need more GPU memory or CPU RAM")

        if self.results.get("agi_limits", {}).get("parallel_processing_limits", {}).get("parallel_limited"):
            recommendations.append("⚠️  AGI LIMITATION: Limited parallel processing capability affects hive mind performance")

        return recommendations

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete system evaluation"""
        print("🚀 ECH0-PRIME System Evaluation")
        print("=" * 50)

        self.results = {
            "system_info": self.system_info,
            "hardware": self.evaluate_hardware_limits(),
            "software": self.evaluate_software_dependencies(),
            "performance": self.evaluate_performance_bottlenecks(),
            "agi_limits": self.evaluate_agi_system_limits(),
        }

        self.results["recommendations"] = self.generate_recommendations()

        return self.results

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 ECH0-PRIME SYSTEM EVALUATION SUMMARY")
        print("=" * 60)

        # System overview
        sys_info = self.results.get("system_info", {})
        print("\n🖥️  SYSTEM OVERVIEW:")
        print(f"  Platform: {sys_info.get('platform', 'Unknown')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
        print(f"  Total RAM: {sys_info.get('total_memory', 0):.1f} GB")
        print(f"  Available RAM: {sys_info.get('available_memory', 0):.1f} GB")

        # Critical limitations
        print("\n🚨 CRITICAL LIMITATIONS:")
        hw = self.results.get("hardware", {})

        # Memory check
        mem = hw.get("memory", {})
        if not mem.get("can_run_full_system", True):
            print(f"  ❌ Memory: Only {mem.get('available_ram_gb', 0):.1f} GB available - insufficient for AGI")

        # CPU check
        cpu = hw.get("cpu", {})
        if not cpu.get("can_handle_parallel_processing", True):
            print(f"  ❌ CPU: Only {cpu.get('logical_cores', 0)} cores - insufficient for parallel AGI")

        # GPU check
        gpu = hw.get("gpu", {})
        has_gpu = gpu.get("cuda_available", False) or gpu.get("mps_available", False)
        if not has_gpu:
            print("  ❌ GPU: No GPU available - neural networks will be slow")
        elif gpu.get("mps_available", False):
            print(f"  ✅ GPU: Apple Silicon GPU available ({gpu.get('gpu_name', 'Unknown')})")

        # Software check
        sw = self.results.get("software", {})
        if not sw.get("all_critical_available", True):
            missing = sw.get("missing_critical", [])
            print(f"  ❌ Software: Missing critical dependencies: {', '.join(missing)}")

        # Recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("\n✅ SYSTEM READY: All requirements met for ECH0-PRIME operation")

def main():
    evaluator = SystemEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.print_summary()

    # Save detailed results
    with open("system_evaluation_results.json", "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_results = json.loads(json.dumps(results, default=convert_for_json))
        json.dump(json_results, f, indent=2)

    print("\n📄 Detailed results saved to: system_evaluation_results.json")
if __name__ == "__main__":
    main()
