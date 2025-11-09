import os
from pathlib import Path

def show_structure():
    """Display codebase structure with line counts."""
    
    radiate_dir = Path("radiate")
    
    print("\nRADIATE CODEBASE STRUCTURE")
    print("="*60)
    
    for file in sorted(radiate_dir.glob("*.py")):
        if file.name == "__init__.py":
            continue
        
        with open(file) as f:
            lines = len(f.readlines())
        
        print(f"\n{file.name} ({lines} lines)")
        print("-" * 40)
        
        # Show main classes and functions
        with open(file) as f:
            for line in f:
                if line.startswith("class ") or line.startswith("def "):
                    print(f"  {line.strip()}")

if __name__ == "__main__":
    show_structure()
