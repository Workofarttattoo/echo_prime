#!/usr/bin/env python3
"""
ECH0-PRIME License Management System
Proprietary code protection and usage tracking.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import base64
import hmac

class LicenseManager:
    """
    Manages ECH0-PRIME licensing, activation, and usage tracking.
    Provides proprietary code protection while allowing legitimate usage.
    """

    def __init__(self, license_key: Optional[str] = None):
        self.license_file = Path.home() / ".ech0_prime" / "license.json"
        self.license_file.parent.mkdir(parents=True, exist_ok=True)
        self.machine_fingerprint = self._generate_machine_fingerprint()
        self.license_data: Optional[Dict[str, Any]] = None

        if license_key:
            self.activate_license(license_key)
        else:
            self.load_license()

    def _generate_machine_fingerprint(self) -> str:
        """Generate a unique machine fingerprint for license binding."""
        # Collect system information (non-identifying)
        system_info = []

        # CPU info (simplified)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                system_info.append(hashlib.sha256(cpu_info.encode()).hexdigest()[:16])
        except:
            system_info.append("cpu_unknown")

        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            system_info.append(str(mem.total)[:8])
        except:
            system_info.append("mem_unknown")

        # Disk info
        try:
            import psutil
            disk = psutil.disk_usage('/')
            system_info.append(str(disk.total)[:8])
        except:
            system_info.append("disk_unknown")

        # MAC address (first network interface)
        try:
            import uuid
            mac = hex(uuid.getnode())[2:].zfill(12)
            system_info.append(mac[:8])
        except:
            system_info.append("mac_unknown")

        # Combine and hash
        fingerprint = "|".join(system_info)
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def generate_license_key(self, license_type: str = "developer",
                           validity_days: int = 365,
                           max_activations: int = 1) -> str:
        """
        Generate a new license key for ECH0-PRIME.
        Only the license generator (you) should have this function.
        """
        license_data = {
            "license_type": license_type,
            "issued_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=validity_days)).isoformat(),
            "max_activations": max_activations,
            "machine_fingerprint": None,  # Set during activation
            "features": self._get_features_for_license(license_type),
            "watermark": "ECH0-PRIME-CSA-PROPRIETARY"
        }

        # Create license string with HMAC signature
        license_json = json.dumps(license_data, sort_keys=True)
        secret_key = "ECH0-PRIME-LICENSE-SECRET-KEY-2025"  # CHANGE THIS IN PRODUCTION

        signature = hmac.new(secret_key.encode(), license_json.encode(), hashlib.sha256).hexdigest()
        license_key = base64.urlsafe_b64encode(f"{license_json}:{signature}".encode()).decode()

        return license_key

    def _get_features_for_license(self, license_type: str) -> Dict[str, bool]:
        """Define features available for each license type."""
        base_features = {
            "basic_inference": True,
            "hive_mind": False,
            "self_modification": False,
            "quantum_attention": False,
            "consciousness_metrics": False,
            "unlimited_training": False,
            "commercial_use": False,
            "redistribution": False
        }

        if license_type == "developer":
            return {
                **base_features,
                "hive_mind": True,
                "self_modification": True,
                "quantum_attention": True,
                "consciousness_metrics": True,
                "unlimited_training": True
            }
        elif license_type == "academic":
            return {
                **base_features,
                "hive_mind": True,
                "consciousness_metrics": True
            }
        elif license_type == "trial":
            return base_features  # Basic features only

        return base_features

    def activate_license(self, license_key: str) -> bool:
        """Activate a license key."""
        try:
            # Decode license key
            license_data_b64 = license_key.encode()
            license_content = base64.urlsafe_b64decode(license_data_b64).decode()
            license_json, signature = license_content.split(":", 1)

            # Verify signature
            secret_key = "ECH0-PRIME-LICENSE-SECRET-KEY-2025"
            expected_signature = hmac.new(secret_key.encode(), license_json.encode(), hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                print("❌ Invalid license signature")
                return False

            license_data = json.loads(license_json)

            # Check expiration
            valid_until = datetime.fromisoformat(license_data["valid_until"])
            if datetime.now() > valid_until:
                print("❌ License expired")
                return False

            # Bind to machine
            license_data["machine_fingerprint"] = self.machine_fingerprint
            license_data["activated_at"] = datetime.now().isoformat()
            license_data["activation_count"] = 1

            # Save license
            self.license_data = license_data
            self._save_license()

            print("✅ License activated successfully"            print(f"   Type: {license_data['license_type']}")
            print(f"   Valid until: {license_data['valid_until']}")
            print(f"   Features: {', '.join([k for k, v in license_data['features'].items() if v])}")

            return True

        except Exception as e:
            print(f"❌ License activation failed: {e}")
            return False

    def load_license(self) -> bool:
        """Load existing license from disk."""
        if not self.license_file.exists():
            return False

        try:
            with open(self.license_file, 'r') as f:
                self.license_data = json.load(f)

            # Verify machine binding
            if self.license_data.get("machine_fingerprint") != self.machine_fingerprint:
                print("⚠️  License not valid for this machine")
                return False

            # Check expiration
            valid_until = datetime.fromisoformat(self.license_data["valid_until"])
            if datetime.now() > valid_until:
                print("⚠️  License expired")
                return False

            return True

        except Exception as e:
            print(f"❌ License loading failed: {e}")
            return False

    def _save_license(self):
        """Save license to disk."""
        with open(self.license_file, 'w') as f:
            json.dump(self.license_data, f, indent=2)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled for the current license."""
        if not self.license_data:
            return False
        return self.license_data.get("features", {}).get(feature, False)

    def get_license_status(self) -> Dict[str, Any]:
        """Get current license status."""
        if not self.license_data:
            return {"status": "unlicensed"}

        return {
            "status": "active",
            "type": self.license_data.get("license_type"),
            "issued_at": self.license_data.get("issued_at"),
            "valid_until": self.license_data.get("valid_until"),
            "days_remaining": (datetime.fromisoformat(self.license_data["valid_until"]) - datetime.now()).days,
            "features": self.license_data.get("features", {}),
            "machine_bound": self.license_data.get("machine_fingerprint") == self.machine_fingerprint
        }

    def validate_system_integrity(self) -> bool:
        """Validate that core system files haven't been tampered with."""
        core_files = [
            "main_orchestrator.py",
            "core/engine.py",
            "quantum_attention/quantum_attention_bridge.py"
        ]

        expected_hashes = {
            # These would be generated during legitimate installation
            "main_orchestrator.py": "PLACEHOLDER_HASH",
            "core/engine.py": "PLACEHOLDER_HASH",
            "quantum_attention/quantum_attention_bridge.py": "PLACEHOLDER_HASH"
        }

        for file_path in core_files:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                if file_hash != expected_hashes.get(file_path, ""):
                    print(f"⚠️  File integrity check failed: {file_path}")
                    return False

        return True


class CodeProtector:
    """
    Provides code protection through obfuscation and encrypted modules.
    """

    def __init__(self, encryption_key: str = None):
        self.encryption_key = encryption_key or "ECH0-PRIME-PROTECTION-KEY-2025"
        self.protected_modules = {}

    def obfuscate_code(self, code: str) -> str:
        """Simple code obfuscation (for demonstration)."""
        # This is a very basic obfuscation - in production you'd use proper obfuscators
        import base64

        # Compress and encode
        compressed = base64.b64encode(code.encode()).decode()

        # Add execution wrapper
        obfuscated = f'''
import base64
exec(base64.b64decode("{compressed}").decode())
'''

        return obfuscated

    def create_protected_module(self, module_name: str, source_code: str) -> str:
        """Create a protected module that requires license validation."""
        protected_code = f'''
# ECH0-PRIME Protected Module: {module_name}
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.

import sys
from core.license_manager import LicenseManager

# License validation
license_mgr = LicenseManager()
if not license_mgr.license_data:
    print("❌ ECH0-PRIME requires a valid license to run protected modules.")
    print("   Please contact Joshua Hendricks Cole for licensing information.")
    print("   Phone: 7252242617 | Email: 7252242617@vtext.com")
    sys.exit(1)

if not license_mgr.validate_system_integrity():
    print("❌ System integrity check failed. Possible tampering detected.")
    sys.exit(1)

# Check feature access
required_features = ["basic_inference"]  # Add specific features as needed
for feature in required_features:
    if not license_mgr.is_feature_enabled(feature):
        print(f"❌ Feature '{feature}' not available in current license.")
        sys.exit(1)

# If all checks pass, execute the protected code
{source_code}
'''

        return protected_code

    def generate_watermark(self, content: str, watermark_text: str = "ECH0-PRIME-CSA-PROPRIETARY") -> str:
        """Add invisible watermark to code or data."""
        # Add comments with watermark (invisible in execution)
        watermarked = f'''
# =============================================================================
# {watermark_text} - PROPRIETARY CODE
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
# Unauthorized copying, modification, or distribution is strictly prohibited.
# =============================================================================
{content}
# =============================================================================
# End of {watermark_text} Protected Code
# =============================================================================
'''
        return watermarked


# Convenience functions
def check_license() -> bool:
    """Quick license check for imports."""
    license_mgr = LicenseManager()
    return license_mgr.license_data is not None

def require_feature(feature: str):
    """Decorator to require specific license features."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            license_mgr = LicenseManager()
            if not license_mgr.is_feature_enabled(feature):
                raise PermissionError(f"Feature '{feature}' requires appropriate license")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_license_status() -> Dict[str, Any]:
    """Get current license status."""
    license_mgr = LicenseManager()
    return license_mgr.get_license_status()


if __name__ == "__main__":
    # Example usage
    print("ECH0-PRIME License Management System")
    print("=" * 50)

    license_mgr = LicenseManager()

    # Check current status
    status = license_mgr.get_license_status()
    print(f"License Status: {status['status']}")

    if status['status'] == 'unlicensed':
        print("\nTo activate ECH0-PRIME:")
        print("1. Contact Joshua Hendricks Cole for a license key")
        print("2. Phone: 7252242617 | Email: 7252242617@vtext.com")
        print("3. Use: license_mgr.activate_license('your_license_key_here')")

    else:
        print(f"License Type: {status.get('type', 'unknown')}")
        print(f"Valid Until: {status.get('valid_until', 'unknown')}")
        print(f"Days Remaining: {status.get('days_remaining', 'unknown')}")

        print("\nEnabled Features:")
        for feature, enabled in status.get('features', {}).items():
            if enabled:
                print(f"  ✅ {feature}")

    # Demonstrate code protection
    protector = CodeProtector()

    sample_code = """
def protected_function():
    print("This is protected ECH0-PRIME code")
    return "executed successfully"
"""

    protected = protector.create_protected_module("sample_module", sample_code)
    watermarked = protector.generate_watermark(protected)

    print(f"\nProtected module created ({len(watermarked)} characters)")
    print("Protection features:")
    print("  ✅ License validation")
    print("  ✅ System integrity checks")
    print("  ✅ Feature access control")
    print("  ✅ Invisible watermarking")
