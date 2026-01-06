import torch
from core.engine import HierarchicalGenerativeModel
import inspect

print(f"File: {inspect.getfile(HierarchicalGenerativeModel)}")
print(f"Signature: {inspect.signature(HierarchicalGenerativeModel.__init__)}")

try:
    model = HierarchicalGenerativeModel(use_cuda=False, lightweight=True)
    print("✅ Successfully instantiated with lightweight=True")
except TypeError as e:
    print(f"❌ Failed to instantiate: {e}")

