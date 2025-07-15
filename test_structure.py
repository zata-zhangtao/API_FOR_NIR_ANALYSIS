#!/usr/bin/env python3
"""
Simple test to verify the package structure is correct.
This test checks if the package can be imported without external dependencies.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_package_structure():
    """Test that the package can be imported correctly."""
    try:
        # Test basic package import
        print("Testing basic package import...")
        import nirapi
        print("✓ nirapi package imported successfully")
        
        # Test version info
        print(f"✓ Package version: {nirapi.__version__}")
        print(f"✓ Package author: {nirapi.__author__}")
        print(f"✓ Package description: {nirapi.__description__}")
        
        # Test AnalysisClass subpackage (this will fail without dependencies, but that's expected)
        print("Testing AnalysisClass subpackage...")
        try:
            from nirapi import AnalysisClass
            print("✓ AnalysisClass subpackage imported successfully")
        except ImportError as e:
            print(f"⚠ AnalysisClass requires dependencies: {e}")
            print("  This is expected behavior - dependencies will be installed during pip install")
        
        print("\n✅ Package structure is correct!")
        print("\nTo install with pip from git, use:")
        print("pip install git+https://github.com/your-username/your-repo.git")
        print("\nOr for development:")
        print("pip install -e git+https://github.com/your-username/your-repo.git#egg=nirapi")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_package_structure()
    sys.exit(0 if success else 1) 