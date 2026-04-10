#!/usr/bin/env python3
"""
ECH0-PRIME Autonomous Repository Evaluation CLI
Real code execution - not simulations
"""

import sys
import argparse
from pathlib import Path

# Add the code_evaluation directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_coder import AutonomousCoder

def main():
    parser = argparse.ArgumentParser(
        description="ECH0-PRIME Autonomous Repository Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_repo.py https://github.com/user/repo.git
  python evaluate_repo.py github.com/user/repo --verbose
  python evaluate_repo.py https://github.com/user/repo.git --no-push
        """
    )

    parser.add_argument(
        'repo_url',
        help='GitHub repository URL to evaluate and improve'
    )

    parser.add_argument(
        '--workspace',
        default=os.environ.get('ECH0_WORKSPACE', os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        help='Workspace directory for cloning and evaluation'
    )

    parser.add_argument(
        '--no-push',
        action='store_true',
        help='Do not push improvements back to repository'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress information'
    )

    args = parser.parse_args()

    # Validate GitHub URL
    if not args.repo_url.startswith(('http', 'https')):
        if 'github.com' in args.repo_url:
            args.repo_url = f"https://{args.repo_url}.git"
        else:
            args.repo_url = f"https://github.com/{args.repo_url}.git"

    if 'github.com' not in args.repo_url:
        print("❌ Error: Only GitHub repositories are supported")
        sys.exit(1)

    print("🤖 ECH0-PRIME Autonomous Repository Evaluation")
    print("=" * 55)
    print(f"🎯 Target: {args.repo_url}")
    print(f"📁 Workspace: {args.workspace}")
    print(f"🚀 Push Changes: {'No' if args.no_push else 'Yes'}")
    print()

    # Initialize autonomous coder
    coder = AutonomousCoder(args.workspace)

    # Override push behavior if requested
    if args.no_push:
        original_push = coder._push_improvements
        coder._push_improvements = lambda *args, **kwargs: {"pushed": False, "message": "Push disabled by --no-push flag"}

    # Perform evaluation
    print("🔄 Starting autonomous evaluation...")
    result = coder.evaluate_github_repo(args.repo_url)

    # Display results
    if "error" in result:
        print(f"❌ Evaluation failed: {result['error']}")
        sys.exit(1)

    print("\n✅ EVALUATION COMPLETE")
    print(f"📦 Repository: {result['repo_url']}")
    print(f"📁 Local Path: {result['local_path']}")
    print()

    # Analysis results
    analysis = result.get('analysis', {})
    if analysis:
        print("🔍 CODE ANALYSIS:")
        if analysis.get('languages'):
            langs = ", ".join([f"{lang}: {count}" for lang, count in analysis['languages'].items()])
            print(f"• Languages: {langs}")

        if analysis.get('security_issues'):
            print(f"• Security Issues: {len(analysis['security_issues'])}")
            for issue in analysis['security_issues'][:3]:
                print(f"  - {issue}")

        if analysis.get('test_coverage', 0) > 0:
            print(f"• Test Coverage: {analysis['test_coverage']:.1f}%")

        print()

    # Improvements
    improvements = result.get('improvements_identified', 0)
    changes = result.get('changes_made', [])

    print("🔧 IMPROVEMENTS:")
    print(f"• Opportunities Identified: {improvements}")
    print(f"• Changes Implemented: {len(changes)}")

    if changes:
        print("• Changes Made:")
        for change in changes:
            print(f"  ✓ {change}")

    print()

    # Test results
    test_results = result.get('test_results', {})
    print("🧪 VALIDATION:")
    if test_results.get('validation_passed'):
        print("• ✅ Syntax validation passed")
    else:
        print(f"• ❌ {len(test_results.get('syntax_errors', []))} syntax errors")

    if test_results.get('tests_run'):
        print(f"• Tests: {test_results.get('tests_passed', 0)} passed, {test_results.get('tests_failed', 0)} failed")
    else:
        print("• Tests: Not run (no test framework detected)")

    print()

    # Push results
    push_result = result.get('push_result', {})
    print("🚀 DEPLOYMENT:")
    if push_result.get('pushed'):
        print(f"• ✅ Changes pushed to branch: {push_result['branch']}")
        print(f"• 📋 Commit: {push_result.get('commit_hash', 'N/A')[:8]}")
    else:
        print("• ℹ️ Changes committed locally (not pushed)")

    print()
    print("🎯 Autonomous evaluation and improvement cycle complete!")
    print("📊 Real code analysis, improvements, and validation performed.")

    if args.verbose:
        print(f"\n📋 Full Results: {result}")

if __name__ == "__main__":
    main()
