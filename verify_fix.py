
with open('training/ensemble_model.py', 'r') as f:
    lines = f.readlines()
    
print("Checking line 405 (evaluate_ensemble):")
print(f"  {lines[404].strip()}")

print("\nChecking line 406 (evaluate_ensemble):")
print(f"  {lines[405].strip()}")

print("\nChecking line 490 (optimize_ensemble_weights):")
print(f"  {lines[489].strip()}")

print("\nChecking line 491 (optimize_ensemble_weights):")
print(f"  {lines[490].strip()}")

print("\n" + "="*50)
if "'label'" in lines[404] or "'label'" in lines[405] or "'label'" in lines[489] or "'label'" in lines[490]:
    print("❌ ERROR: File still contains 'label' references!")
else:
    print("✅ SUCCESS: File correctly uses 'class' column!")
print("="*50)
