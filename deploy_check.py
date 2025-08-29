#!/usr/bin/env python3
"""
Deployment verification script for Render
Checks all dependencies and imports work correctly
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        # Test basic dependencies
        import streamlit
        print("✓ streamlit")
        
        import langchain
        print("✓ langchain")
        
        import langchain_community
        print("✓ langchain_community")
        
        import langchain_openai
        print("✓ langchain_openai")
        
        import langchain_chroma
        print("✓ langchain_chroma")
        
        import langchain_google_community
        print("✓ langchain_google_community")
        
        import chromadb
        print("✓ chromadb")
        
        import openai
        print("✓ openai")
        
        import PyPDF2
        print("✓ PyPDF2")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_rag_imports():
    """Test RAG-specific imports"""
    print("\n🔍 Testing RAG imports...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Test RAG imports
        from agents.rag import (
            SourceType, 
            SourcePreference, 
            EnhancedRAGSystem
        )
        print("✓ RAG components imported successfully")
        
        print("🎉 RAG imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ RAG import error: {e}")
        return False
    except Exception as e:
        print(f"❌ RAG unexpected error: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("\n🔍 Checking environment variables...")
    
    required_vars = ['OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if os.environ.get(var):
            print(f"✓ {var} is set")
        else:
            print(f"⚠ {var} is not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠ Missing environment variables: {', '.join(missing_vars)}")
        print("These will need to be set in Render dashboard")
        return False
    else:
        print("🎉 All environment variables are set!")
        return True

def main():
    """Main verification function"""
    print("🚀 Render Deployment Verification")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test RAG imports
    if not test_rag_imports():
        success = False
    
    # Check environment
    if not check_environment():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All checks passed! Ready for deployment.")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
