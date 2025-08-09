import os
import sys
import json
import re
import readline
import atexit
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import shlex
import glob

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm_handler import get_available_models, get_models_by_framework, MODEL_REGISTRY
from core.train import train_from_prompt
from core.tune import tune_from_prompt
from core.logs import show_logs

class EpochInteractiveCLI:
    """Interactive command-line interface for Epoch ML toolkit"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.session_data = {
            "models_used": set(),
            "datasets_used": set(),
            "last_model": None,
            "last_dataset": None,
            "command_count": 0,
            "deployed_models": []
        }
        
        # Setup command history
        self.history_file = os.path.expanduser("~/.epoch_history")
        self.setup_readline()
        
        # IMPROVED: More flexible command patterns
        self.command_patterns = {
            'train': [
                r'train\s+(?:a\s+|an\s+)?(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$',
                r'(?:build|create|fit)\s+(?:a\s+|an\s+)?(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$',
                r'run\s+(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'tune': [
                r'tune\s+(?:hyperparameters?\s+(?:for\s+)?|(?:a\s+|an\s+)?)?(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$',
                r'optimize\s+(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$',
                r'hpo\s+(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'compare': [
                r'compare\s+(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$',
                r'benchmark\s+(.+?)(?:\s+on\s+(.+?))?(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'deploy': [
                r'deploy\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'(?:create|generate)\s+(?:fastapi\s+)?(?:api|deployment)\s+(?:for\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'serve\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$'
            ]
        }
        
        # Available datasets (can be extended)
        self.known_datasets = {
            'iris', 'wine', 'breast_cancer', 'digits', 'diabetes', 'titanic',
            'boston', 'california_housing', 'auto_mpg', 'heart', 'mushroom', 'penguins'
        }
        
        # Command suggestions
        available_models = get_available_models()
        self.suggestions = {
            'models': list(available_models),
            'datasets': list(self.known_datasets),
            'frameworks': list(set(config['framework'] for config in MODEL_REGISTRY.values())),
            'commands': ['train', 'tune', 'compare', 'benchmark', 'deploy', 'list', 'help', 'history', 'clear', 'exit']
        }

    def setup_readline(self):
        """Setup readline for command history and tab completion"""
        try:
            # Load command history
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
            
            # Set history length
            readline.set_history_length(1000)
            
            # Setup tab completion
            readline.set_completer(self.complete)
            readline.parse_and_bind('tab: complete')
            
            # Save history on exit
            atexit.register(self.save_history)
            
        except ImportError:
            print("⚠️  readline not available - history and tab completion disabled")

    def save_history(self):
        """Save command history to file"""
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass

    def complete(self, text: str, state: int) -> Optional[str]:
        """Enhanced tab completion handler with context awareness"""
        try:
            line = readline.get_line_buffer()
            matches = []
            words = line.lower().split()
            
            # Complete commands at the beginning
            if not line.strip() or line.strip() == text:
                matches = [cmd for cmd in self.suggestions['commands'] if cmd.startswith(text)]
            else:
                # Context-aware completion based on previous words
                
                # After train/tune/optimize commands, suggest models
                if any(word in ['train', 'tune', 'optimize', 'build', 'create', 'fit'] for word in words):
                    matches.extend([model for model in self.suggestions['models'] if model.startswith(text)])
                
                # After deploy command, suggest trained model files
                if 'deploy' in words:
                    # Look for .joblib and .pkl files in current directory and outputs/
                    model_files = []
                    for pattern in ['*.joblib', '*.pkl', 'outputs/*.joblib', 'outputs/*.pkl']:
                        model_files.extend(glob.glob(pattern))
                    # Remove directory prefix for cleaner completion
                    clean_files = [os.path.basename(f) for f in model_files if os.path.basename(f).startswith(text)]
                    matches.extend(clean_files)
                
                # After 'on' or dataset keywords, suggest datasets
                if any(word in ['on', 'dataset', 'data', 'using'] for word in words[-2:]):
                    matches.extend([dataset for dataset in self.suggestions['datasets'] if dataset.startswith(text)])
                
                # After 'framework', suggest frameworks
                if 'framework' in words:
                    matches.extend([fw for fw in self.suggestions['frameworks'] if fw.startswith(text)])
                
                # After 'compare', suggest model combinations
                if 'compare' in words:
                    # Suggest individual models
                    matches.extend([model for model in self.suggestions['models'] if model.startswith(text)])
                    # Suggest common model combinations
                    common_combos = [
                        'random_forest,xgboost,lightgbm',
                        'sklearn models',
                        'tree models',
                        'ensemble models'
                    ]
                    matches.extend([combo for combo in common_combos if combo.startswith(text)])
                
                # After 'port', suggest common port numbers
                if 'port' in words:
                    common_ports = ['8000', '8080', '5000', '3000', '9000']
                    matches.extend([port for port in common_ports if port.startswith(text)])
                
                # Suggest common phrases and options
                common_phrases = [
                    'with cv', 'with cross-validation', 'with 5-fold cv', 'with 10-fold cv',
                    'with 50 trials', 'with 100 trials', 'framework sklearn', 'framework xgboost',
                    'stratified cv', 'kfold cv', 'port 8000', 'port 8080', 'to deployment_fastapi'
                ]
                matches.extend([phrase for phrase in common_phrases if phrase.startswith(text)])
                
                # Smart dataset suggestions based on recent usage
                if self.session_data['last_dataset'] and self.session_data['last_dataset'].startswith(text):
                    matches.insert(0, self.session_data['last_dataset'])
                
                # Smart model suggestions based on recent usage
                if self.session_data['last_model'] and self.session_data['last_model'].startswith(text):
                    matches.insert(0, self.session_data['last_model'])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_matches = []
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    unique_matches.append(match)
            
            if state < len(unique_matches):
                return unique_matches[state]
            else:
                return None
                
        except Exception:
            return None

    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*70)
        print("🤖 Welcome to Epoch CLI v{} - Interactive ML Command Shell".format(self.version))
        print("="*70)
        print("🎯 Train, tune, compare, and deploy ML models with natural language commands")
        print("📚 Type 'help' for available commands and examples")
        print("🔄 Use arrow keys for command history, Tab for auto-completion")
        print("💡 Example: 'train a random forest on penguins with 10-fold cv'")
        print("🚀 Example: 'deploy model.joblib to my_api port 8080'")
        print("❌ Type 'exit' or 'quit' to leave")
        print("="*70 + "\n")

    def print_help(self):
        """Print help information"""
        help_text = """
🔧 EPOCH INTERACTIVE CLI COMMANDS

📊 TRAINING COMMANDS:
  train <model> on <dataset> [options]     Train a model
  tune <model> on <dataset> [options]      Hyperparameter optimization
  compare <models> on <dataset> [options]  Compare multiple models
  benchmark <dataset> [options]            Comprehensive benchmarking

🚀 DEPLOYMENT COMMANDS:
  deploy <model_file> [to <output_dir>] [port <port>]    Generate FastAPI app
  deploy model.joblib                                    Deploy to default directory
  deploy model.pkl to my_api port 8080                   Custom deployment

🛠️  UTILITY COMMANDS:
  list models [framework]                   Show available models
  list datasets                            Show known datasets
  list deployments                         Show deployed models
  help                                     Show this help
  history                                  Show command history
  clear                                    Clear screen
  exit / quit                              Exit the CLI

🎯 NATURAL LANGUAGE EXAMPLES:
  • train a random forest on penguins
  • tune xgboost on titanic with 100 trials
  • compare random_forest,xgboost,svm on wine with cv
  • benchmark iris with sklearn models
  • train lgbm on breast_cancer with 5-fold cv
  • optimize lightgbm hyperparameters on diabetes
  • deploy trained_model.joblib to my_fastapi_app port 9000
  • create fastapi api for model.pkl

⚙️  OPTIONS (can be mixed into commands):
  • with cv / with cross-validation         Enable cross-validation
  • with N-fold cv                          Specify CV folds
  • with N trials                           Set HPO trials
  • framework sklearn/xgboost/lightgbm      Filter by framework
  • stratified cv / kfold cv                CV strategy
  • port NNNN                              Specify deployment port
  • to <directory>                         Specify output directory

📁 AVAILABLE MODELS: {}

🗂️  KNOWN DATASETS: {}
        """.format(
            ', '.join(sorted(self.suggestions['models'])),
            ', '.join(sorted(self.suggestions['datasets']))
        )
        print(help_text)

    def parse_natural_language(self, command: str) -> Tuple[str, Dict]:
        """IMPROVED: Parse natural language command into action and parameters"""
        command = command.strip().lower()
        
        # Handle simple commands first
        if command in ['help', 'h', '?']:
            return 'help', {}
        elif command in ['exit', 'quit', 'q']:
            return 'exit', {}
        elif command in ['clear', 'cls']:
            return 'clear', {}
        elif command in ['history', 'hist']:
            return 'history', {}
        elif command.startswith('list'):
            return 'list', {'args': command.split()[1:]}
        
        # Parse complex training commands
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if action == 'deploy':
                        # Special handling for deploy command
                        model_file = groups[0] if groups[0] else ""
                        output_dir = groups[1] if len(groups) > 1 and groups[1] else "deployment_fastapi"
                        port = int(groups[2]) if len(groups) > 2 and groups[2] else 8000
                        
                        params = {
                            'action': 'deploy',
                            'model_file': model_file.strip(),
                            'output_dir': output_dir.strip() if output_dir else "deployment_fastapi",
                            'port': port
                        }
                        return action, params
                    else:
                        # Extract model, dataset, and options
                        model_part = groups[0] if groups[0] else ""
                        dataset_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        options_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        
                        # Clean and parse components
                        params = self.extract_parameters(command, model_part, dataset_part, options_part)
                        params['action'] = action
                        
                        return action, params
        
        # If no pattern matches, try to extract key information
        return 'unknown', {'original_command': command}

    def extract_parameters(self, full_command: str, model_part: str, dataset_part: str, options_part: str) -> Dict:
        """IMPROVED: Extract parameters from command components with better model list handling"""
        params = {}
        
        # FIXED: Extract models (handle comma-separated lists better)
        if model_part:
            model_part = model_part.strip()
            
            # Check if this looks like a comma-separated list of models
            if ',' in model_part:
                # Split by comma and clean each model name
                model_candidates = [m.strip() for m in model_part.split(',')]
                valid_models = []
                available_models = get_available_models()
                
                for candidate in model_candidates:
                    # Clean the candidate name
                    clean_candidate = re.sub(r'\b(a|an|the|model|classifier|regressor)\b', '', candidate, flags=re.IGNORECASE).strip()
                    clean_candidate = re.sub(r'\s+', '_', clean_candidate)
                    
                    # Try to match with available models
                    best_match = None
                    best_score = 0
                    
                    for model in available_models:
                        # Exact match
                        if model.lower() == clean_candidate.lower():
                            best_match = model
                            best_score = 1.0
                            break
                        # Partial match
                        elif clean_candidate.lower() in model.lower() or model.lower() in clean_candidate.lower():
                            score = len(clean_candidate) / max(len(model), len(clean_candidate))
                            if score > best_score:
                                best_match = model
                                best_score = score
                    
                    if best_match and best_score > 0.5:
                        valid_models.append(best_match)
                    else:
                        # Try the original candidate as-is
                        valid_models.append(clean_candidate)
                
                params['models'] = valid_models
            else:
                # Single model - existing logic
                # Remove articles and common words
                model_part = re.sub(r'\b(a|an|the|model|classifier|regressor)\b', '', model_part, flags=re.IGNORECASE).strip()
                model_part = re.sub(r'\s+', '_', model_part)

                # Try to match with available models (fuzzy matching)
                available_models = get_available_models()
                best_match = None
                best_score = 0
                
                for model in available_models:
                    # Exact match
                    if model.lower() == model_part.lower():
                        best_match = model
                        best_score = 1.0
                        break
                    # Partial match
                    elif model_part.lower() in model.lower() or model.lower() in model_part.lower():
                        score = len(model_part) / max(len(model), len(model_part))
                        if score > best_score:
                            best_match = model
                            best_score = score
                
                if best_match and best_score > 0.5:
                    params['model'] = best_match
                else:
                    # If no good match, use the cleaned model part
                    params['model'] = model_part
        
        # Handle framework specification (e.g., "sklearn models", "xgboost models")
        if model_part and ('models' in model_part or 'framework' in full_command):
            framework_match = re.search(r'(sklearn|xgboost|lightgbm|tensorflow|pytorch|catboost)\s+models?', model_part, re.IGNORECASE)
            if framework_match:
                params['framework'] = framework_match.group(1).lower()
                # Remove the model specification since we're using framework
                if 'model' in params:
                    del params['model']
                if 'models' in params:
                    del params['models']
        
        # IMPROVED: Extract dataset with better matching
        if dataset_part:
            dataset_part = dataset_part.strip()
            # Remove common words but be more careful
            dataset_part = re.sub(r'\b(the|dataset|data)\b', '', dataset_part, flags=re.IGNORECASE).strip()
            # Remove extra whitespace
            dataset_part = re.sub(r'\s+', ' ', dataset_part).strip()
            
            # Try to match with known datasets
            best_match = None
            best_score = 0
            
            for dataset in self.known_datasets:
                # Exact match
                if dataset.lower() == dataset_part.lower():
                    best_match = dataset
                    best_score = 1.0
                    break
                # Partial match
                elif dataset_part.lower() in dataset.lower() or dataset.lower() in dataset_part.lower():
                    score = len(dataset_part) / max(len(dataset), len(dataset_part))
                    if score > best_score:
                        best_match = dataset
                        best_score = score
            
            if best_match and best_score >= 0.7:  # Higher threshold for datasets
                params['dataset'] = best_match
            else:
                # Use the cleaned dataset part as-is
                params['dataset'] = dataset_part
        
        # Extract cross-validation settings
        cv_match = re.search(r'(\d+)[-\s]*fold\s+(?:cv|cross.?validation)', full_command, re.IGNORECASE)
        if cv_match:
            params['cv_folds'] = int(cv_match.group(1))
            params['use_cv'] = True
        elif re.search(r'\b(?:cv|cross.?validation)\b', full_command, re.IGNORECASE):
            params['use_cv'] = True
            params['cv_folds'] = 5  # default
        
        # Extract CV strategy
        if re.search(r'stratified', full_command, re.IGNORECASE):
            params['cv_type'] = 'stratified'
        elif re.search(r'kfold|k.fold', full_command, re.IGNORECASE):
            params['cv_type'] = 'kfold'
        
        # Extract number of trials for hyperparameter optimization
        trials_match = re.search(r'(\d+)\s+trials?', full_command, re.IGNORECASE)
        if trials_match:
            params['n_trials'] = int(trials_match.group(1))
        
        # Extract framework filter
        framework_match = re.search(r'framework\s+(\w+)', full_command, re.IGNORECASE)
        if framework_match:
            params['framework'] = framework_match.group(1)
        
        # Extract deployment-specific parameters
        port_match = re.search(r'port\s+(\d+)', full_command, re.IGNORECASE)
        if port_match:
            params['port'] = int(port_match.group(1))
        
        output_dir_match = re.search(r'(?:to|as)\s+([^\s]+)', full_command, re.IGNORECASE)
        if output_dir_match:
            params['output_dir'] = output_dir_match.group(1)
        
        return params

    def validate_and_enhance_params(self, action: str, params: Dict) -> Dict:
        """Validate parameters and prompt for missing required ones"""
        enhanced_params = params.copy()
        
        if action in ['train', 'tune']:
            # Check for required model
            if not enhanced_params.get('model'):
                print("🤔 No model specified.")
                model = self.prompt_for_model()
                if model:
                    enhanced_params['model'] = model
                else:
                    print("❌ Model selection cancelled")
                    return {}
            
            # Check for required dataset
            if not enhanced_params.get('dataset'):
                print("🤔 No dataset specified.")
                dataset = self.prompt_for_dataset()
                if dataset:
                    enhanced_params['dataset'] = dataset
                else:
                    print("❌ Dataset selection cancelled")
                    return {}
                    
            # Validate model exists ONLY if it's not already in available models
            available_models = get_available_models()
            if enhanced_params['model'] not in available_models:
                print(f"⚠️  Model '{enhanced_params['model']}' not recognized.")
                suggestions = [m for m in available_models if enhanced_params['model'].lower() in m.lower()]
                if suggestions:
                    print(f"💡 Similar models: {', '.join(suggestions[:3])}")
                    choice = input("Use the first suggestion? (y/n): ").lower()
                    if choice == 'y':
                        enhanced_params['model'] = suggestions[0]
                        print(f"✅ Using {suggestions[0]}")
                    else:
                        print("❌ Model validation failed")
                        return {}
                else:
                    print("❌ No similar models found")
                    return {}
        
        elif action == 'compare':
            # Check for required dataset
            if not enhanced_params.get('dataset'):
                print("🤔 No dataset specified for comparison.")
                dataset = self.prompt_for_dataset()
                if dataset:
                    enhanced_params['dataset'] = dataset
                else:
                    print("❌ Dataset selection cancelled")
                    return {}
        
        elif action == 'deploy':
            # Check for required model file
            if not enhanced_params.get('model_file'):
                print("🤔 No model file specified.")
                model_file = self.prompt_for_model_file()
                if model_file:
                    enhanced_params['model_file'] = model_file
                else:
                    print("❌ Model file selection cancelled")
                    return {}
            
            # Validate model file exists
            model_file = enhanced_params['model_file']
            
            # Try different possible paths
            possible_paths = [
                model_file,
                os.path.join('outputs', model_file),
                f"{model_file}.joblib" if not model_file.endswith(('.joblib', '.pkl')) else model_file,
                os.path.join('outputs', f"{model_file}.joblib") if not model_file.endswith(('.joblib', '.pkl')) else os.path.join('outputs', model_file)
            ]
            
            valid_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    valid_path = path
                    break
            
            if valid_path:
                enhanced_params['model_file'] = valid_path
                print(f"✅ Found model file: {valid_path}")
            else:
                print(f"❌ Model file not found: {model_file}")
                print("💡 Available model files:")
                model_files = glob.glob('*.joblib') + glob.glob('*.pkl') + glob.glob('outputs/*.joblib') + glob.glob('outputs/*.pkl')
                if model_files:
                    for i, f in enumerate(model_files[:5], 1):  # Show first 5
                        print(f"  {i}. {f}")
                    if len(model_files) > 5:
                        print(f"  ... and {len(model_files) - 5} more")
                else:
                    print("  No model files found in current directory or outputs/")
                return {}
        
        return enhanced_params

    def handle_train_command(self, params: Dict):
        """Handle training commands"""
        # Validate and enhance parameters first
        params = self.validate_and_enhance_params('train', params)
        if not params:
            return
            
        model = params.get('model')
        dataset = params.get('dataset')
        
        # Build prompt for existing train function
        prompt = f"model: {model} dataset: {dataset}"
        
        # Add CV configuration
        cv_config = {}
        if params.get('use_cv'):
            cv_config['use_cv'] = True
            cv_config['cv_folds'] = params.get('cv_folds', 5)
            cv_config['cv_type'] = params.get('cv_type', 'auto')
            
            cv_info = f" with {cv_config['cv_folds']}-fold CV"
            print(f"🎯 Training {model} on {dataset}{cv_info}")
        else:
            cv_config['use_cv'] = False
            print(f"🎯 Training {model} on {dataset}")
        
        # Update session data
        self.session_data['models_used'].add(model)
        self.session_data['datasets_used'].add(dataset)
        self.session_data['last_model'] = model
        self.session_data['last_dataset'] = dataset
        self.session_data['command_count'] += 1
        
        # Call existing train function
        try:
            train_from_prompt(prompt, cv_config=cv_config)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Check if the model and dataset names are correct")

    def handle_tune_command(self, params: Dict):
        """Handle hyperparameter tuning commands with validation"""
        # Validate and enhance parameters
        params = self.validate_and_enhance_params('tune', params)
        if not params:
            return
            
        model = params.get('model')
        dataset = params.get('dataset')
        n_trials = params.get('n_trials', 50)
        
        # Build prompt for existing tune function
        prompt = f"model: {model} dataset: {dataset} n_trials: {n_trials}"
        
        print(f"🔬 Optimizing {model} hyperparameters on {dataset} ({n_trials} trials)")
        
        # Update session data
        self.session_data['models_used'].add(model)
        self.session_data['datasets_used'].add(dataset)
        self.session_data['last_model'] = model
        self.session_data['last_dataset'] = dataset
        self.session_data['command_count'] += 1
        
        # Call existing tune function
        try:
            tune_from_prompt(prompt)
        except Exception as e:
            print(f"❌ Tuning failed: {e}")
            print("💡 Check if the model and dataset names are correct")

    def handle_compare_command(self, params: Dict):
        """Handle model comparison commands with validation"""
        # Validate and enhance parameters
        params = self.validate_and_enhance_params('compare', params)
        if not params:
            return
            
        models = params.get('models', [])
        dataset = params.get('dataset')
        framework = params.get('framework')
        
        if not models and not framework:
            # Use default comparison models
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
            print(f"🏁 Using default models for comparison: {', '.join(models)}")
        
        # Import and call compare function from cli.py
        try:
            from cli import compare
            
            prompt = f"dataset: {dataset}"
            
            # Set up arguments
            kwargs = {
                'prompt': prompt,
                'models': ','.join(models) if models else None,
                'framework': framework,
                'save_results': True,
                'use_cv': params.get('use_cv', False),
                'cv_folds': params.get('cv_folds', 5),
                'generate_viz': True
            }
            
            print(f"🏁 Comparing models on {dataset}")
            if params.get('use_cv'):
                print(f"📊 Using {params.get('cv_folds', 5)}-fold cross-validation")
            
            # Update session data
            if models:
                self.session_data['models_used'].update(models)
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
            compare(**kwargs)
            
        except ImportError:
            print("❌ Compare function not available")
        except Exception as e:
            print(f"❌ Comparison failed: {e}")

    def handle_deploy_command(self, params: Dict):
        """Handle deployment commands with validation"""
        # Validate and enhance parameters
        params = self.validate_and_enhance_params('deploy', params)
        if not params:
            return
            
        model_file = params.get('model_file')
        output_dir = params.get('output_dir', 'deployment_fastapi')
        port = params.get('port', 8000)
        
        print(f"🚀 Deploying model: {model_file}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🌐 Port: {port}")
        
        # Import and call fastapi deployer
        try:
            # Import the fastapi_deployer function
            sys.path.append('core')
            from core.fastapi_deployer import generate_fastapi_app
            
            # Generate the FastAPI deployment
            generate_fastapi_app(model_file, output_dir, port)
            
            # Update session data
            deployment_info = {
                'model_file': model_file,
                'output_dir': output_dir,
                'port': port,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.session_data['deployed_models'].append(deployment_info)
            self.session_data['command_count'] += 1
            
            print(f"✅ FastAPI deployment created successfully!")
            print(f"📂 Files created in: {output_dir}/")
            print(f"🔧 Next steps:")
            print(f"   cd {output_dir}")
            print(f"   pip install -r requirements.txt")
            print(f"   python main.py")
            print(f"🌍 API will be available at: http://localhost:{port}")
            print(f"📖 API docs at: http://localhost:{port}/docs")
            
        except ImportError as e:
            print(f"❌ Deployment failed: FastAPI deployer not found")
            print("💡 Make sure fastapi_deployer.py exists in the core/ directory")
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            print("💡 Check if the model file is valid and readable")
    
    def handle_compare_command(self, params: Dict):
        """Handle model comparison commands with validation"""
        # Validate and enhance parameters
        params = self.validate_and_enhance_params('compare', params)
        if not params:
            return
            
        models = params.get('models', [])
        dataset = params.get('dataset')
        framework = params.get('framework')
        
        if not models and not framework:
            # Use default comparison models
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
            print(f"🏁 Using default models for comparison: {', '.join(models)}")
        
        # Import and call compare function from cli.py
        try:
            from cli import compare
            
            prompt = f"dataset: {dataset}"
            
            # Set up arguments
            kwargs = {
                'prompt': prompt,
                'models': ','.join(models) if models else None,
                'framework': framework,
                'save_results': True,
                'use_cv': params.get('use_cv', False),
                'cv_folds': params.get('cv_folds', 5),
                'generate_viz': True
            }
            
            print(f"🏁 Comparing models on {dataset}")
            if params.get('use_cv'):
                print(f"📊 Using {params.get('cv_folds', 5)}-fold cross-validation")
            
            # Update session data
            if models:
                self.session_data['models_used'].update(models)
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
            compare(**kwargs)
            
        except ImportError:
            print("❌ Compare function not available")
        except Exception as e:
            print(f"❌ Comparison failed: {e}")

    def handle_advisor_command(self, params: Dict):
        """FIXED: Handle model advisor commands with proper validation"""
        # Validate and enhance parameters first
        params = self.validate_and_enhance_params('advisor', params)
        if not params:
            return
            
        dataset = params.get('dataset')
        
        print(f"🧠 Getting intelligent recommendations for {dataset}")
        
        try:
            from cli import advisor
            advisor(
                dataset=dataset,  # Pass as positional argument
                detailed=params.get('detailed', True),
                auto_compare=params.get('auto_compare', False),
                save_report=True,
                prefer_interpretable=params.get('prefer_interpretable', False),
                prefer_fast=params.get('prefer_fast', False)
            )
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
        except ImportError:
            print("❌ Advisor function not available")
        except Exception as e:
            print(f"❌ Advisor failed: {e}")
            print("💡 Make sure the dataset name is correct")
            print("💡 Available datasets: iris, wine, breast_cancer, penguins, etc.")

    def prompt_for_model(self) -> Optional[str]:
        """Interactive model selection"""
        available_models = get_available_models()
        frameworks = {}
        
        # Group models by framework
        for model in available_models:
            fw = MODEL_REGISTRY[model]["framework"]
            if fw not in frameworks:
                frameworks[fw] = []
            frameworks[fw].append(model)
        
        print("\n📊 Available models by framework:")
        model_choices = []
        idx = 1
        
        for fw, models in frameworks.items():
            print(f"\n🔧 {fw.upper()}:")
            for model in models:
                print(f"  {idx:2d}. {model}")
                model_choices.append(model)
                idx += 1
        
        print(f"\n  {idx:2d}. Skip (cancel)")
        
        try:
            choice = input(f"\nSelect a model (1-{len(model_choices)}): ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_choices):
                    return model_choices[choice_idx]
            elif choice.lower() in available_models:
                return choice.lower()
        except (ValueError, KeyboardInterrupt):
            pass
        
        return None
    
    def prompt_for_dataset(self) -> Optional[str]:
        """Interactive dataset selection"""
        print("\n🗂️  Common datasets:")
        datasets = sorted(list(self.known_datasets))
        
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i:2d}. {dataset}")
        
        print(f"  {len(datasets) + 1:2d}. Enter custom dataset name")
        print(f"  {len(datasets) + 2:2d}. Skip (cancel)")
        
        try:
            choice = input(f"\nSelect a dataset (1-{len(datasets) + 2}): ").strip()
            
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(datasets):
                    return datasets[choice_idx]
                elif choice_idx == len(datasets):
                    # Custom dataset
                    custom = input("Enter dataset name: ").strip()
                    return custom if custom else None
            elif choice.lower() in self.known_datasets:
                return choice.lower()
        except (ValueError, KeyboardInterrupt):
            pass
        
        return None

    def execute_command(self, action: str, params: Dict) -> bool:
        """Execute the parsed command"""
        try:
            if action == 'help':
                self.print_help()
            
            elif action == 'exit':
                return False
            
            elif action == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
            
            elif action == 'history':
                self.show_history()
            
            elif action == 'list':
                self.handle_list_command(params.get('args', []))
            
            elif action == 'train':
                self.handle_train_command(params)
            
            elif action == 'tune':
                self.handle_tune_command(params)
            
            elif action == 'compare':
                self.handle_compare_command(params)

            elif action == 'advisor':
                self.handle_advisor_command(params)
            
            elif action == 'deploy':
                self.handle_deploy_command(params)

            elif action == 'unknown':
                original_cmd = params.get('original_command', '')
                print(f"❓ Couldn't understand command: '{original_cmd}'")
                
                # Try to suggest corrections
                suggestions = self.suggest_corrections(original_cmd)
                if suggestions:
                    print("💡 Did you mean:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
                    
                    # Allow user to select a suggestion
                    try:
                        choice = input("Select a suggestion (1-{}) or press Enter to skip: ".format(len(suggestions)))
                        if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                            corrected_command = suggestions[int(choice) - 1]
                            print(f"🔄 Executing: {corrected_command}")
                            action, params = self.parse_natural_language(corrected_command)
                            return self.execute_command(action, params)
                    except (ValueError, KeyboardInterrupt):
                        pass
                
                print("💡 Type 'help' for available commands and examples")
            
            else:
                print(f"⚠️  Command '{action}' not implemented yet")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing command: {e}")
            return True

    def suggest_corrections(self, command: str) -> List[str]:
        """Suggest corrections for unknown commands using fuzzy matching"""
        suggestions = []
        command_lower = command.lower()
        
        # Check for common typos and variations
        typo_corrections = {
    'trian': 'train', 'tarni': 'train', 'traing': 'train',
    'optimzie': 'optimize', 'opitmize': 'optimize',
    'comapre': 'compare', 'comprae': 'compare',
    'benchamrk': 'benchmark', 'bencmark': 'benchmark',
    'ramdom': 'random', 'forst': 'forest',
    'xgbost': 'xgboost', 'xgboost': 'xgboost',
    'lgbm': 'lightgbm', 'lightbgm': 'lightgbm',
    'analze': 'analyze', 'analize': 'analyze',
    'advisr': 'advisor', 'advise': 'advisor',
    'depoy': 'deploy', 'deplpy': 'deploy', 'deply': 'deploy',
    'delpoy': 'deploy', 'depooy': 'deploy', 'depliy': 'deploy'
}

        
        # Check for direct typo corrections
        for typo, correction in typo_corrections.items():
            if typo in command_lower:
                corrected = command_lower.replace(typo, correction)
                suggestions.append(corrected)
        
        # Simple fuzzy matching for commands
        all_suggestions = (self.suggestions['commands'] + 
                          self.suggestions['models'] + 
                          self.suggestions['datasets'])
        
        for suggestion in all_suggestions:
            # Check if suggestion is similar (simple edit distance)
            if self.simple_similarity(command_lower, suggestion.lower()) > 0.6:
                suggestions.append(suggestion)
        
        return list(set(suggestions))[:3]  # Return top 3 unique suggestions
    
    def simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings"""
        if not s1 or not s2:
            return 0.0
        
        # Check for substring matches
        if s1 in s2 or s2 in s1:
            return 0.8
        
        # Simple character-based similarity
        common_chars = set(s1) & set(s2)
        total_chars = set(s1) | set(s2)
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)

    def handle_list_command(self, args: List[str]):
        """Handle list commands"""
        if not args:
            print("📋 Available list options: models, datasets, frameworks, history")
            return
        
        command = args[0].lower()
        
        if command == 'models':
            framework_filter = args[1] if len(args) > 1 else None
            self.list_models(framework_filter)
        elif command == 'datasets':
            self.list_datasets()
        elif command == 'frameworks':
            self.list_frameworks()
        elif command == 'history':
            self.show_history()
        else:
            print(f"❓ Unknown list command: {command}")

    def list_models(self, framework_filter: Optional[str] = None):
        """List available models"""
        if framework_filter:
            models = get_models_by_framework(framework_filter)
            if models:
                print(f"📊 Models for {framework_filter}:")
                for model in models:
                    print(f"  • {model}")
            else:
                print(f"❌ No models found for framework: {framework_filter}")
        else:
            print("📊 All Available Models:")
            
            # Group by framework
            frameworks = {}
            for model_name in get_available_models():
                fw = MODEL_REGISTRY[model_name]["framework"]
                if fw not in frameworks:
                    frameworks[fw] = []
                frameworks[fw].append(model_name)
            
            for fw, models in frameworks.items():
                print(f"\n🔧 {fw.upper()}:")
                for model in models:
                    print(f"  • {model}")

    def list_datasets(self):
        """List known datasets"""
        print("🗂️  Known Datasets:")
        for dataset in sorted(self.known_datasets):
            print(f"  • {dataset}")

    def list_frameworks(self):
        """List available frameworks"""
        frameworks = set(config['framework'] for config in MODEL_REGISTRY.values())
        print("🔧 Available Frameworks:")
        for fw in sorted(frameworks):
            print(f"  • {fw}")

    def show_history(self):
        """Show recent command history"""
        try:
            history_length = readline.get_current_history_length()
            if history_length == 0:
                print("📜 No command history available")
                return
            
            print("📜 Recent Commands:")
            # Show last 10 commands
            start = max(1, history_length - 9)
            for i in range(start, history_length + 1):
                try:
                    cmd = readline.get_history_item(i)
                    if cmd:
                        print(f"  {i:2d}: {cmd}")
                except Exception:
                    pass
        except NameError:
            print("📜 Command history not available (readline not loaded)")

    def show_session_stats(self):
        """Show current session statistics"""
        if self.session_data['command_count'] > 0:
            print(f"\n📊 Session Stats:")
            print(f"  Commands run: {self.session_data['command_count']}")
            if self.session_data['models_used']:
                print(f"  Models used: {', '.join(sorted(self.session_data['models_used']))}")
            if self.session_data['datasets_used']:
                print(f"  Datasets used: {', '.join(sorted(self.session_data['datasets_used']))}")
            if self.session_data['last_model']:
                print(f"  Last model: {self.session_data['last_model']}")
            if self.session_data['last_dataset']:
                print(f"  Last dataset: {self.session_data['last_dataset']}")

    def run(self):
        """Main interactive loop"""
        self.print_banner()
        
        try:
            while True:
                try:
                    # Get command with prompt
                    command = input("epoch> ").strip()
                    
                    if not command:
                        continue
                    
                    # Parse and execute command
                    action, params = self.parse_natural_language(command)
                    
                    if not self.execute_command(action, params):
                        break  # Exit requested
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Use 'exit' to quit gracefully")
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
        
        finally:
            self.show_session_stats()
            print("\n🎉 Thanks for using Epoch CLI!")


def main():
    """Entry point for interactive CLI"""
    cli = EpochInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()