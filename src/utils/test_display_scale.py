"""
Test script for DisplayScaleManager
"""
from display_scale_manager import DisplayScaleManager

print("="*60)
print("Display Scale Manager - Test Script")
print("="*60)

# Test 1: Get current scale
print("\n[Test 1] Getting current display scale...")
current = DisplayScaleManager.get_current_scale()
print(f"✓ Current scale: {current}%")

# Test 2: Get available scales
print("\n[Test 2] Getting available scales...")
available = DisplayScaleManager.get_available_scales()
print(f"✓ Available scales: {available}")

# Test 3: Set display scale (CAUTION: This will change your display!)
print("\n[Test 3] Testing set_display_scale()...")
print("⚠️  WARNING: This will attempt to change your display scale!")
print("    You may need to log out/in for full effect.")

response = input("\nDo you want to test changing the scale? (yes/no): ")

if response.lower() == 'yes':
    test_scale = input(f"Enter scale to test {available}: ")
    try:
        test_scale = int(test_scale)
        print(f"\nAttempting to set scale to {test_scale}%...")
        success, message = DisplayScaleManager.set_display_scale(test_scale)

        if success:
            print(f"✓ SUCCESS: {message}")
            print("\nVerifying change...")
            new_scale = DisplayScaleManager.get_current_scale()
            print(f"✓ Scale now reads as: {new_scale}%")

            if new_scale == test_scale:
                print("✓ Change confirmed!")
            else:
                print("⚠️  Registry updated, but may need logout/login to take effect")
        else:
            print(f"✗ FAILED: {message}")
    except ValueError:
        print("✗ Invalid input - must be a number")
else:
    print("\nSkipping scale change test.")

print("\n" + "="*60)
print("Test complete!")
print("="*60)

