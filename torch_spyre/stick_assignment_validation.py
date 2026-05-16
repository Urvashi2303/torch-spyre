"""
Simple end-to-end test for stick alignment validation

Tests that misaligned tensors are caught by the validator.
Similar to examples/mul.py but tests the NEGATIVE case.
"""
import os
import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

print("=" * 70)
print("Testing Stick Alignment Validation")
print("=" * 70)

# Create MISALIGNED tensors (1023 NOT divisible by 64)
print("\n[1] Creating misaligned tensors...")
print("    Shape: [128, 1023]")
print("    1023 / 64 = 15.984 (PARTIAL STICK - INVALID!)")

x = torch.rand(128, 1023, dtype=torch.float16)
y = torch.rand(128, 1023, dtype=torch.float16)

x_device = x.to(DEVICE)
y_device = y.to(DEVICE)

print("\n[2] Attempting to compile and run...")
print("    Expected: Validation error about stick alignment")

try:
    compiled = torch.compile(lambda a, b: torch.mul(a, b))
    result = compiled(x_device, y_device)
    
    print("\n  TEST FAILED: Validator did not catch misalignment!")
    print(f"   Operation executed and returned: {result.shape}")
    
except (ValueError, RuntimeError) as e:
    print(f"\n✅ Validation error raised (as expected):")
    print(f"   {str(e)[:200]}...")
    print("\n" + "=" * 70)
    print("✅ TEST PASSED: Stick alignment validation works!")
    print("=" * 70)
    print("\nWhat this prevents:")
    print("  • Without validation: Code compiles but CRASHES on hardware")
    print("  • With validation: Error caught early with fix suggestion")
