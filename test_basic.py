#!/usr/bin/env python3
"""
Basic functionality test for the RAG system
Tests core components without requiring API keys
"""
import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic imports that don't require API keys"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test standard library imports
        import json
        import re
        import warnings
        from typing import Dict, List, Tuple, Optional
        from enum import Enum
        print("✓ Standard library imports")
        
        # Test basic package imports
        import streamlit
        print("✓ Streamlit")
        
        # Test LangChain imports
        import langchain
        import langchain_core
        print("✓ LangChain core")
        
        print("🎉 Basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "agents/ui.py",
        "agents/rag.py", 
        "requirements.txt",
        "documents/"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("🎉 All required files present!")
        return True

def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("\n🔍 Testing Python syntax...")
    
    python_files = ["agents/ui.py", "agents/rag.py"]
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} syntax valid")
        except SyntaxError as e:
            print(f"❌ {file_path} syntax error: {e}")
            syntax_errors.append(file_path)
        except Exception as e:
            print(f"❌ {file_path} error: {e}")
            syntax_errors.append(file_path)
    
    if syntax_errors:
        print(f"\n❌ Syntax errors in: {', '.join(syntax_errors)}")
        return False
    else:
        print("🎉 All Python files have valid syntax!")
        return True

def main():
    """Main test function"""
    print("🚀 Basic Functionality Test")
    print("=" * 40)
    
    success = True
    
    # Test basic imports
    if not test_basic_imports():
        success = False
    
    # Test file structure
    if not test_file_structure():
        success = False
    
    # Test Python syntax
    if not test_python_syntax():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All basic tests passed!")
        print("Next: Set environment variables and test with deploy_check.py")
        return 0
    else:
        print("❌ Some basic tests failed. Please fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
