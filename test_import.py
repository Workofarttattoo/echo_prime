try:
    from reasoning.orchestrator import ChainOfThoughtReasoner
    print("✅ ChainOfThoughtReasoner imported successfully")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Other Error: {e}")
    import traceback
    traceback.print_exc()
