#!/usr/bin/env python3
"""
Epoch Interactive Mode Launcher
Usage: python epoch.py [--interactive | -i]
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Epoch ML Toolkit")
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Start interactive command shell')
    parser.add_argument('--version', action='version', version='Epoch CLI v1.0.0')
    
    # If no arguments provided, show help and suggest interactive mode
    if len(sys.argv) == 1:
        print("🤖 Epoch ML Toolkit")
        print("Usage:")
        print("  python epoch.py -i              # Start interactive shell")
        print("  python epoch.py --interactive   # Start interactive shell")
        print("  python cli.py <command>         # Run direct commands")
        print("\nFor direct commands, use: python cli.py --help")
        print("For interactive mode with natural language, use: python epoch.py -i")
        return
    
    args = parser.parse_args()
    
    if args.interactive:
        # Import and start interactive CLI
        try:
            from interactive_cli import EpochInteractiveCLI
            cli = EpochInteractiveCLI()
            cli.run()
        except ImportError as e:
            print(f"❌ Could not import interactive CLI: {e}")
            print("Make sure interactive_cli.py is in the same directory")
        except Exception as e:
            print(f"❌ Error starting interactive CLI: {e}")
    else:
        print("Use -i or --interactive to start the interactive shell")

if __name__ == "__main__":
    main()