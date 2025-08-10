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
    """Enhanced Interactive command-line interface for Epoch ML toolkit with better NLP"""
    
    def __init__(self):
        self.version = "1.1.0"
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
        
        # Enhanced command patterns with more flexible dataset extraction
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
  
            'deploy': [
                r'deploy\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'(?:create|generate)\s+(?:fastapi\s+)?(?:api|deployment)\s+(?:for\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'serve\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$'

            'advisor': [
                # More flexible advisor patterns
                r'(?:analyze|advisor?|recommend|suggest)\s+(?:models?\s+(?:for\s+)?)?(.+?)(?:\s+dataset)?(?:\s+(?:with|using)\s+(.+))?$',
                r'what\s+(?:models?|algorithms?)\s+(?:should\s+i\s+use\s+(?:for\s+)?|work\s+best\s+(?:for\s+)?)?(.+?)(?:\s+dataset)?(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'engineer': [
                # Better feature engineering patterns
                r'engineer\s+features?\s+(?:for\s+)?(.+?)(?:\s+(?:with|using)\s+(.+))?',
                r'create\s+features?\s+(?:for\s+)?(.+?)(?:\s+(?:with|using)\s+(.+))?',
                r'generate\s+features?\s+(?:for\s+)?(.+?)(?:\s+(?:with|using)\s+(.+))?',
                r'build\s+features?\s+(?:for\s+)?(.+?)(?:\s+(?:with|using)\s+(.+))?',
                r'(?:create|engineer|generate)\s+(?:polynomial|mathematical|statistical)\s+features?\s+(?:for\s+)?(.+?)(?:\s+(?:with|using)\s+(.+))?'
            ],
            'select': [
                # Enhanced feature selection patterns
                r'select\s+(?:top\s+)?(\d+)?\s*features?\s+(?:from\s+)?(.+?)(?:\s+(?:using|with|method)\s+(.+))?',
                r'choose\s+(?:best\s+)?(\d+)?\s*features?\s+(?:from\s+)?(.+?)(?:\s+(?:using|with|method)\s+(.+))?',
                r'pick\s+(?:top\s+)?(\d+)?\s*features?\s+(?:from\s+)?(.+?)(?:\s+(?:using|with|method)\s+(.+))
            ]
        }
        
        # Available datasets (expanded)
        self.known_datasets = {
            'iris', 'wine', 'breast_cancer', 'digits', 'diabetes', 'titanic',
            'boston', 'california_housing', 'auto_mpg', 'heart', 'mushroom', 'penguins',
            'tips', 'flights', 'mpg', 'cars', 'boston_housing', 'california', 'housing'
        }
        
        # Command suggestions
        available_models = get_available_models()
        self.suggestions = {
            'models': list(available_models),
            'datasets': list(self.known_datasets),
            'frameworks': list(set(config['framework'] for config in MODEL_REGISTRY.values())),

            'commands': ['train', 'tune', 'compare', 'benchmark', 'deploy','advisor', 'analyze', 'engineer', 'select', 'list', 'help', 'history', 'clear', 'exit']

        }

    def setup_readline(self):
        """Setup readline for command history and tab completion"""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
            readline.set_history_length(1000)
            readline.set_completer(self.complete)
            readline.parse_and_bind('tab: complete')
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
        """Enhanced tab completion with better context awareness"""
        try:
            line = readline.get_line_buffer()
            matches = []
            words = line.lower().split()
            
            if not line.strip() or line.strip() == text:
                matches = [cmd for cmd in self.suggestions['commands'] if cmd.startswith(text)]
            else:
                # Context-aware completion
                if any(word in ['train', 'tune', 'optimize', 'build', 'create', 'fit'] for word in words):
                    matches.extend([model for model in self.suggestions['models'] if model.startswith(text)])
                

                # After deploy command, suggest trained model files
                if 'deploy' in words or '--model_file' in words:
                    # Look for .joblib and .pkl files in current directory and outputs/
                    model_files = []
                    for pattern in ['*.joblib', '*.pkl', 'outputs/*.joblib', 'outputs/*.pkl', 'outputs/models/*.joblib', 'outputs/models/*.pkl']:
                        model_files.extend(glob.glob(pattern))
                    # Remove directory prefix for cleaner completion and add quotes for paths with spaces
                    clean_files = []
                    for f in model_files:
                        basename = os.path.basename(f)
                        if basename.startswith(text) or f.startswith(text):
                            # Add quotes if path contains spaces
                            display_path = f'"{f}"' if ' ' in f else f
                            clean_files.append(display_path)
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
                if 'port' in words or '--port' in words:
                    common_ports = ['8000', '8080', '5000', '3000', '9000']
                    matches.extend([port for port in common_ports if port.startswith(text)])
                
                # Suggest common phrases and options
                common_phrases = [
                    'with cv', 'with cross-validation', 'with 5-fold cv', 'with 10-fold cv',
                    'with 50 trials', 'with 100 trials', 'framework sklearn', 'framework xgboost',
                    'stratified cv', 'kfold cv', 'port 8000', 'port 8080', 'to deployment_fastapi',
                    '--model_file', '--output_dir', '--port', '--output', '-o', '-p'
                ]
                matches.extend([phrase for phrase in common_phrases if phrase.startswith(text)])

                if any(word in ['on', 'dataset', 'data', 'from', 'for', 'using'] for word in words[-3:]):
                    matches.extend([dataset for dataset in self.suggestions['datasets'] if dataset.startswith(text)])
                
                if any(word in ['analyze', 'advisor', 'recommend', 'suggest'] for word in words):
                    matches.extend([dataset for dataset in self.suggestions['datasets'] if dataset.startswith(text)])

                
                # Smart suggestions based on recent usage
                if self.session_data['last_dataset'] and self.session_data['last_dataset'].startswith(text):
                    matches.insert(0, self.session_data['last_dataset'])
                
                if self.session_data['last_model'] and self.session_data['last_model'].startswith(text):
                    matches.insert(0, self.session_data['last_model'])
            
            # Remove duplicates
            seen = set()
            unique_matches = []
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    unique_matches.append(match)
            
            return unique_matches[state] if state < len(unique_matches) else None
                
        except Exception:
            return None

    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*70)
        print("🤖 Enhanced Epoch CLI v{} - Smart ML Command Shell".format(self.version))
        print("="*70)
        print("🎯 Train, tune, compare, and deploy ML models with natural language commands")
        print("🔄 Use arrow keys for command history, Tab for auto-completion")
        print("💡 Example: 'train a random forest on penguins with 10-fold cv'")
        print("🚀 Example: 'deploy model.joblib to my_api port 8080'")
        print("🏷️  Example: 'deploy --model_file model.pkl --output_dir my_api --port 8080'")
        print("📚 Type 'help' for commands or try these examples:")
        print("   • train random forest on penguins")
        print("   • select features from wine dataset")
        print("   • analyze iris dataset")
        print("   • engineer features for titanic")

        print("❌ Type 'exit' or 'quit' to leave")
        print("="*70 + "\n")

    def print_help(self):
        """Print help information"""
        help_text = """
🔧 EPOCH INTERACTIVE CLI COMMANDS

📊 TRAINING COMMANDS:
  train <model> on <dataset>               Train a model
  tune <model> on <dataset>                Optimize hyperparameters  
  compare <models> on <dataset>            Compare multiple models
  benchmark <dataset>                      Comprehensive benchmarking

🧠 ANALYSIS COMMANDS:
  analyze <dataset>                        Get intelligent model recommendations
  advisor <dataset>                        Detailed model suggestions
  recommend models for <dataset>           Quick model recommendations

🔧 FEATURE ENGINEERING:
  engineer features for <dataset>          Comprehensive feature engineering
  create polynomial features for <dataset> Polynomial/interaction features
  select features from <dataset>           Feature selection
  select top N features from <dataset>     Select specific number of features

🚀 DEPLOYMENT COMMANDS (Natural Language):
  deploy <model_file> [to <output_dir>] [port <port>]    Generate FastAPI app
  deploy model.joblib                                    Deploy to default directory
  deploy model.pkl to my_api port 8080                   Custom deployment

🚀 DEPLOYMENT COMMANDS (CLI Style):
  deploy --model_file <file> --output_dir <dir> --port <port>
  deploy --model_file "path/to/model.pkl" --output my_api --port 8080

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
  • select features from wine dataset
  • analyze penguins
  • engineer features for titanic
  • compare xgboost and random forest on iris
  • tune lightgbm on breast cancer
  • select top 20 features from diabetes

🏷️  CLI STYLE EXAMPLES:
  • deploy --model_file model.pkl --output_dir my_api --port 8080
  • deploy --model_file "G:\\path\\to\\model.pkl" --output my_api
  • deploy --model_file outputs/model.joblib --port 9000

⚙️  OPTIONS (can be mixed into commands):
  • with cv / with cross-validation         Enable cross-validation
  • with N-fold cv                          Specify CV folds
  • with N trials                           Set HPO trials
  • framework sklearn/xgboost/lightgbm      Filter by framework
  • stratified cv / kfold cv                CV strategy
  • port NNNN                              Specify deployment port
  • to <directory>                         Specify output directory
  • --model_file <file>                    CLI-style model file specification
  • --output_dir <dir>                     CLI-style output directory
  • --port <port>                          CLI-style port specification

📁 AVAILABLE MODELS: {}

🗂️  KNOWN DATASETS: {}

        """.format(
            ', '.join(sorted(self.suggestions['models'])),
            ', '.join(sorted(self.suggestions['datasets']))
        )
        print(help_text)


    def parse_cli_flags(self, command: str) -> Tuple[str, Dict]:
        """Parse CLI-style flag commands"""
        try:
            # Parse command line style arguments
            parts = shlex.split(command)
            if not parts:
                return 'unknown', {}
            
            cmd = parts[0].lower()
            
            if cmd == 'deploy':
                params = {'action': 'deploy'}
                i = 1
                
                while i < len(parts):
                    if parts[i].startswith('--'):
                        flag = parts[i][2:]  # Remove '--'
                        if flag == 'model_file' and i + 1 < len(parts):
                            params['model_file'] = parts[i + 1]
                            i += 2
                        elif flag in ['output_dir', 'output'] and i + 1 < len(parts):
                            params['output_dir'] = parts[i + 1]
                            i += 2
                        elif flag == 'port' and i + 1 < len(parts):
                            params['port'] = int(parts[i + 1])
                            i += 2
                        else:
                            i += 1
                    elif parts[i].startswith('-') and not parts[i].startswith('--'):
                        # Handle short flags like -o, -p
                        flag = parts[i][1:]
                        if flag == 'o' and i + 1 < len(parts):
                            params['output_dir'] = parts[i + 1]
                            i += 2
                        elif flag == 'p' and i + 1 < len(parts):
                            params['port'] = int(parts[i + 1])
                            i += 2
                        else:
                            i += 1
                    else:
                        # First non-flag argument is model_file
                        if 'model_file' not in params:
                            params['model_file'] = parts[i]
                        i += 1
                
                return 'deploy', params
        
        except Exception as e:
            print(f"⚠️  Error parsing CLI flags: {e}")
        
        return 'unknown', {}

    def parse_natural_language(self, command: str) -> Tuple[str, Dict]:
        """IMPROVED: Parse natural language command into action and parameters"""
        command = command.strip()
        
        # Check if this looks like a CLI-style command (has -- flags)
        if '--' in command:
            action, params = self.parse_cli_flags(command)
            if action != 'unknown':
                return action, params
        
        # Continue with existing natural language parsing...
        command = command.lower()

    def enhanced_dataset_extraction(self, command: str, context_parts: List[str] = None) -> Optional[str]:
        """Enhanced dataset name extraction with multiple strategies"""
        command_lower = command.lower()
        
        # Strategy 1: Look for explicit dataset mentions with context keywords
        dataset_patterns = [
            r'(?:from|on|for|using|with|analyze|advisor?)\s+(?:the\s+)?(\w+)(?:\s+dataset)?',
            r'dataset\s+(\w+)',
            r'(\w+)\s+dataset',
            r'(?:^|\s)(\w+)(?:\s+(?:data|db))?(?:\s|$)',
        ]
        
        for pattern in dataset_patterns:
            matches = re.findall(pattern, command_lower, re.IGNORECASE)
            for match in matches:
                candidate = match.strip()
                # Skip common words that aren't datasets
                skip_words = {
                    'features', 'feature', 'models', 'model', 'top', 'best', 'using', 'with', 
                    'method', 'select', 'create', 'train', 'tune', 'compare', 'analyze',
                    'polynomial', 'mathematical', 'statistical', 'data', 'dataset',
                    'from', 'for', 'on', 'the', 'and', 'or', 'is', 'are'
                }
                
                if candidate and candidate not in skip_words:
                    # Check if it's a known dataset
                    matched_dataset = self.fuzzy_match_dataset(candidate)
                    if matched_dataset:
                        return matched_dataset
                    # If not found but looks like a dataset name, return as-is
                    elif len(candidate) > 2 and candidate.isalpha():
                        return candidate
        
        # Strategy 2: Check context parts if provided
        if context_parts:
            for part in context_parts:
                if part:
                    matched = self.fuzzy_match_dataset(part)
                    if matched:
                        return matched
        
        # Strategy 3: Look for dataset names anywhere in the command
        words = command_lower.split()
        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word:
                matched = self.fuzzy_match_dataset(clean_word)
                if matched:
                    return matched
        
        return None

    def fuzzy_match_dataset(self, candidate: str) -> Optional[str]:
        """Improved fuzzy matching for dataset names"""
        if not candidate:
            return None
        
        candidate = candidate.lower().strip()
        
        # Exact match first
        if candidate in self.known_datasets:
            return candidate
        
        # Check for partial matches
        for dataset in self.known_datasets:
            # Check if candidate is contained in dataset name
            if candidate in dataset or dataset in candidate:
                # Calculate similarity score
                min_len = min(len(candidate), len(dataset))
                max_len = max(len(candidate), len(dataset))
                similarity = min_len / max_len
                
                if similarity >= 0.6:  # Lower threshold for better matching
                    return dataset
        
        # Check for common dataset aliases
        aliases = {
            'breast': 'breast_cancer',
            'cancer': 'breast_cancer', 
            'penguin': 'penguins',
            'housing': 'california_housing',
            'boston': 'boston',
            'tip': 'tips',
            'car': 'cars',
            'mpg': 'auto_mpg'
        }
        
        for alias, dataset in aliases.items():
            if alias in candidate or candidate in alias:
                return dataset
        
        # If no match found, return original if it looks like a valid dataset name
        if candidate.isalpha() and len(candidate) > 2:
            return candidate
        
        return None

    def enhanced_command_parsing(self, command: str) -> Tuple[str, Dict]:
        """Enhanced command parsing with better NLP understanding"""
        command = command.strip().lower()

        
        # Handle simple commands first
        simple_commands = {
            'help': 'help', 'h': 'help', '?': 'help',
            'exit': 'exit', 'quit': 'exit', 'q': 'exit',
            'clear': 'clear', 'cls': 'clear',
            'history': 'history', 'hist': 'history'
        }
        
        if command in simple_commands:
            return simple_commands[command], {}
        
        if command.startswith('list'):
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
        # Enhanced action detection
        action = self.detect_action(command)

        
        if action == 'unknown':
            return 'unknown', {'original_command': command}
        
        # Extract parameters based on action
        params = self.enhanced_parameter_extraction(command, action)
        params['action'] = action
        
        return action, params

    def detect_action(self, command: str) -> str:
        """Detect the main action from the command"""
        # Action keywords mapping
        action_keywords = {
            'train': ['train', 'fit', 'build', 'run'],
            'tune': ['tune', 'optimize', 'hpo', 'hyperparameter'],
            'compare': ['compare', 'benchmark'],
            'advisor': ['analyze', 'advisor', 'adviser', 'recommend', 'suggest', 'what'],
            'engineer': ['engineer', 'create', 'generate', 'build'],
            'select': ['select', 'choose', 'pick', 'filter']
        }
        
        # Check for feature engineering specific patterns
        if any(keyword in command for keyword in ['features', 'feature']):
            if any(keyword in command for keyword in ['select', 'choose', 'pick', 'top']):
                return 'select'
            elif any(keyword in command for keyword in ['engineer', 'create', 'generate', 'build']):
                return 'engineer'
        
        # Check for model advisor patterns  
        if any(pattern in command for pattern in ['what models', 'which models', 'recommend models', 'suggest models']):
            return 'advisor'
        
        # Check primary action keywords
        for action, keywords in action_keywords.items():
            if any(keyword in command for keyword in keywords):
                return action
        
        return 'unknown'

    def enhanced_parameter_extraction(self, command: str, action: str) -> Dict:
        """Enhanced parameter extraction with better understanding"""
        params = {}
        
        # Extract dataset name using enhanced method
        dataset = self.enhanced_dataset_extraction(command)
        if dataset:
            params['dataset'] = dataset
        
        # Extract model information
        if action in ['train', 'tune', 'compare']:
            model = self.extract_model_names(command)
            if model:
                if isinstance(model, list):
                    params['models'] = model
                else:
                    params['model'] = model
        
        # Extract feature selection parameters
        if action == 'select':
            # Extract number of features
            k_match = re.search(r'(?:top|best|select)\s*(\d+)', command)
            if k_match:
                params['k'] = int(k_match.group(1))
            
            # Extract selection method
            methods = ['univariate', 'recursive', 'model_based', 'correlation', 'variance']
            for method in methods:
                if method in command or method.replace('_', ' ') in command:
                    params['method'] = method
                    break
        
        # Extract CV parameters
        if re.search(r'\d+-?fold|cv|cross.validation', command):
            params['use_cv'] = True
            cv_match = re.search(r'(\d+)-?fold', command)
            if cv_match:
                params['cv_folds'] = int(cv_match.group(1))
        
        # Extract trial count for tuning
        if action == 'tune':
            trials_match = re.search(r'(\d+)\s*trials?', command)
            if trials_match:
                params['n_trials'] = int(trials_match.group(1))
        
        return params

    def extract_model_names(self, command: str) -> Optional[str or List[str]]:
        """Extract model names from command"""
        available_models = get_available_models()
        
        # Look for comma-separated model lists
        if ',' in command:
            # Extract potential model list
            model_part = re.search(r'(?:train|tune|compare)\s+([^on]+?)(?:\s+on|\s*$)', command)
            if model_part:
                model_candidates = [m.strip() for m in model_part.group(1).split(',')]
                matched_models = []
                
                for candidate in model_candidates:
                    cleaned = re.sub(r'\b(and|or|vs|versus|,)\b', '', candidate).strip()
                    cleaned = re.sub(r'\b(a|an|the|model)\b', '', cleaned).strip()
                    matched = self.fuzzy_match_model(cleaned, available_models)
                    if matched:
                        matched_models.append(matched)
                
                return matched_models if matched_models else None
     
        # Extract deployment-specific parameters
        port_match = re.search(r'port\s+(\d+)', full_command, re.IGNORECASE)
        if port_match:
            params['port'] = int(port_match.group(1))
        
        output_dir_match = re.search(r'(?:to|as)\s+([^\s]+)', full_command, re.IGNORECASE)
        if output_dir_match:
            params['output_dir'] = output_dir_match.group(1)
        
        return params

        # Single model extraction
        model_patterns = [
            r'(?:train|tune|fit|build|run)\s+(?:a\s+|an\s+|the\s+)?([^on]+?)(?:\s+on|$)',
            r'(?:model|algorithm):\s*(\w+)',
            r'using\s+(?:a\s+|an\s+|the\s+)?([^on]+?)(?:\s+on|$)'
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                cleaned = re.sub(r'\b(a|an|the|model|classifier|regressor)\b', '', candidate).strip()
                matched = self.fuzzy_match_model(cleaned, available_models)
                if matched:
                    return matched
        
        return None

    def fuzzy_match_model(self, candidate: str, available_models: List[str]) -> Optional[str]:
        """Fuzzy match model names"""
        if not candidate:
            return None
        
        candidate = candidate.lower().strip()
        candidate = re.sub(r'\s+', '_', candidate)  # Replace spaces with underscores
        
        # Exact match
        if candidate in available_models:
            return candidate
        
        # Partial matching
        best_match = None
        best_score = 0
        
        for model in available_models:
            # Check containment in both directions
            if candidate in model.lower():
                score = len(candidate) / len(model)
                if score > best_score:
                    best_match = model
                    best_score = score
            elif model.lower() in candidate:
                score = len(model) / len(candidate)  
                if score > best_score:
                    best_match = model
                    best_score = score
        
        # Return match if confidence is high enough
        return best_match if best_score > 0.4 else None


    def validate_and_enhance_params(self, action: str, params: Dict) -> Dict:
        """Enhanced parameter validation with better user prompts"""
        enhanced_params = params.copy()
        
        # For actions that need datasets
        if action in ['train', 'tune', 'compare', 'advisor', 'engineer', 'select'] and not enhanced_params.get('dataset'):
            print("🤔 No dataset specified.")
            print("💡 Try: 'train random_forest on penguins' or 'analyze iris dataset'")
            
            dataset = self.prompt_for_dataset()
            if dataset:
                enhanced_params['dataset'] = dataset
                print(f"✅ Using dataset: {dataset}")
            else:
                print("❌ Dataset selection cancelled")
                return {}
        
        # For actions that need models  
        if action in ['train', 'tune'] and not enhanced_params.get('model'):
            print("🤔 No model specified.")
            print("💡 Try: 'train random_forest on penguins' or 'tune xgboost on iris'")
            
            model = self.prompt_for_model()
            if model:
                enhanced_params['model'] = model
                print(f"✅ Using model: {model}")
            else:
                print("❌ Model selection cancelled")
                return {}
        
        # Validate model exists
        if enhanced_params.get('model'):
            available_models = get_available_models()
            if enhanced_params['model'] not in available_models:
                print(f"⚠️  Model '{enhanced_params['model']}' not found.")
                # Try to suggest similar models
                similar = [m for m in available_models if enhanced_params['model'].lower() in m.lower()]
                if similar:
                    print(f"💡 Similar models: {', '.join(similar[:3])}")
                    choice = input("Use first suggestion? (y/n): ").lower()
                    if choice == 'y':
                        enhanced_params['model'] = similar[0]
                        print(f"✅ Using {similar[0]}")
                    else:
                        return {}
                else:
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
            
            # Remove quotes if present
            model_file = model_file.strip('"\'')
            
            # Try different possible paths
            possible_paths = [
                model_file,
                os.path.join('outputs', model_file),
                os.path.join('outputs', 'models', model_file),
                f"{model_file}.joblib" if not model_file.endswith(('.joblib', '.pkl')) else model_file,
                os.path.join('outputs', f"{model_file}.joblib") if not model_file.endswith(('.joblib', '.pkl')) else os.path.join('outputs', model_file),
                os.path.join('outputs', 'models', f"{model_file}.joblib") if not model_file.endswith(('.joblib', '.pkl')) else os.path.join('outputs', 'models', model_file)
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
                model_files = (glob.glob('*.joblib') + glob.glob('*.pkl') + 
                              glob.glob('outputs/*.joblib') + glob.glob('outputs/*.pkl') +
                              glob.glob('outputs/models/*.joblib') + glob.glob('outputs/models/*.pkl'))
                if model_files:
                    for i, f in enumerate(model_files[:10], 1):  # Show first 10
                        print(f"  {i}. {f}")
                    if len(model_files) > 10:
                        print(f"  ... and {len(model_files) - 10} more")
                else:
                    print("  No model files found in current directory, outputs/, or outputs/models/")
                return {}
        
        return enhanced_params


    def prompt_for_model_file(self) -> Optional[str]:
        """Interactive model file selection"""
        # Look for model files
        model_files = (glob.glob('*.joblib') + glob.glob('*.pkl') + 
                      glob.glob('outputs/*.joblib') + glob.glob('outputs/*.pkl') +
                      glob.glob('outputs/models/*.joblib') + glob.glob('outputs/models/*.pkl'))
        
        if not model_files:
            print("❌ No model files found")
            custom_path = input("Enter model file path manually: ").strip()
            return custom_path if custom_path else None
        
        print("\n🗂️  Available model files:")
        for i, f in enumerate(model_files, 1):
            print(f"  {i:2d}. {f}")
        
        print(f"  {len(model_files) + 1:2d}. Enter custom path")
        print(f"  {len(model_files) + 2:2d}. Skip (cancel)")
        
        try:
            choice = input(f"\nSelect a model file (1-{len(model_files) + 2}): ").strip()
            
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_files):
                    return model_files[choice_idx]
                elif choice_idx == len(model_files):
                    # Custom path
                    custom_path = input("Enter model file path: ").strip()
                    return custom_path if custom_path else None

    def prompt_for_dataset(self) -> Optional[str]:
        """Interactive dataset selection with better UX"""
        print("\n🗂️  Available datasets:")
        datasets = sorted(list(self.known_datasets))
        
        # Show in columns for better readability
        cols = 3
        for i in range(0, len(datasets), cols):
            row = datasets[i:i+cols]
            print("  " + "".join(f"{j+i+1:2d}. {ds:<20}" for j, ds in enumerate(row)))
        
        print(f"\n  {len(datasets) + 1:2d}. Enter custom dataset")
        print(f"  {len(datasets) + 2:2d}. Cancel")
        
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets) + 2}): ").strip()
            
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(datasets):
                    return datasets[choice_idx]
                elif choice_idx == len(datasets):
                    custom = input("Enter dataset name: ").strip()
                    return custom if custom else None
            elif choice.lower() in self.known_datasets:
                return choice.lower()

        except (ValueError, KeyboardInterrupt):
            pass
        
        return None


    def prompt_for_model(self) -> Optional[str]:
        """Interactive model selection with better UX"""
        available_models = get_available_models()
        frameworks = {}
        
        # Group by framework
        for model in available_models:
            fw = MODEL_REGISTRY[model]["framework"]
            if fw not in frameworks:
                frameworks[fw] = []
            frameworks[fw].append(model)
        
        print("\n📊 Available models:")
        model_choices = []
        idx = 1
        
        for fw, models in frameworks.items():
            print(f"\n🔧 {fw.upper()}:")
            for model in models:
                print(f"  {idx:2d}. {model}")
                model_choices.append(model)
                idx += 1
        
        print(f"\n  {idx:2d}. Cancel")
        
        try:
            choice = input(f"\nSelect model (1-{len(model_choices)}): ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_choices):
                    return model_choices[choice_idx]
        except (ValueError, KeyboardInterrupt):
            pass
        
        return None

    def execute_command(self, action: str, params: Dict) -> bool:
        """Execute the parsed command with enhanced error handling"""
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
            elif action == 'engineer':
                self.handle_engineer_command(params)
            elif action == 'select':
                self.handle_select_command(params)
            elif action == 'unknown':
                self.handle_unknown_command(params)
            else:
                print(f"⚠️  Command '{action}' not implemented yet")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing command: {e}")
            import traceback
            traceback.print_exc()
            return True

    def handle_unknown_command(self, params: Dict):
        """Enhanced handling of unknown commands with better suggestions"""
        original_cmd = params.get('original_command', '')
        print(f"❓ Couldn't understand: '{original_cmd}'")
        
        # Try to suggest corrections
        suggestions = self.get_smart_suggestions(original_cmd)
        if suggestions:
            print("\n💡 Did you mean:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
            
            try:
                choice = input(f"\nSelect suggestion (1-{len(suggestions)}) or Enter to skip: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                    corrected_command = suggestions[int(choice) - 1]
                    print(f"🔄 Executing: {corrected_command}")
                    action, params = self.enhanced_command_parsing(corrected_command)
                    return self.execute_command(action, params)
            except (ValueError, KeyboardInterrupt):
                pass
        else:
            # Provide examples based on detected intent
            self.provide_contextual_examples(original_cmd)

    def get_smart_suggestions(self, command: str) -> List[str]:
        """Generate smart suggestions for unknown commands"""
        suggestions = []
        command_lower = command.lower()
        
        # Common typo corrections
        corrections = {
            'trian': 'train', 'analize': 'analyze', 'selct': 'select',
            'enginer': 'engineer', 'ramdom': 'random', 'forst': 'forest'
        }
        
        for typo, correction in corrections.items():
            if typo in command_lower:
                suggestions.append(command_lower.replace(typo, correction))
        
        # Template-based suggestions
        if 'train' in command_lower or 'fit' in command_lower:
            suggestions.extend([
                "train random_forest on iris",
                "train xgboost on penguins with cv"
            ])
        
        if 'select' in command_lower or 'feature' in command_lower:
            suggestions.extend([
                "select features from penguins dataset",
                "select top 20 features from wine using model_based"
            ])
        
        if 'analyze' in command_lower or 'recommend' in command_lower:
            suggestions.extend([
                "analyze penguins dataset",
                "recommend models for iris"
            ])
        
        return suggestions[:5]  # Limit to top 5

    def provide_contextual_examples(self, command: str):
        """Provide contextual examples based on detected intent"""
        command_lower = command.lower()
        
        print("\n💡 Here are some example commands you can try:")
        
        if any(word in command_lower for word in ['train', 'model', 'fit']):
            print("   🎯 Training: train random_forest on iris")
            print("   🎯 Training: train xgboost on penguins with 10-fold cv")
        
        if any(word in command_lower for word in ['feature', 'select', 'engineer']):
            print("   🔧 Feature Engineering: engineer features for titanic")
            print("   🔧 Feature Selection: select top 50 features from wine")
        
        if any(word in command_lower for word in ['analyze', 'recommend', 'suggest']):
            print("   🧠 Analysis: analyze breast_cancer dataset")
            print("   🧠 Recommendations: recommend models for diabetes")
        
        print("\n📚 Type 'help' for complete command reference")


    def handle_train_command(self, params: Dict):
        """Handle training commands with enhanced validation"""
        params = self.validate_and_enhance_params('train', params)
        if not params:
            return
        
        model = params.get('model')
        dataset = params.get('dataset')
        
        # Build prompt
        prompt = f"model: {model} dataset: {dataset}"
        
        # CV configuration
        cv_config = {
            'use_cv': params.get('use_cv', False),
            'cv_folds': params.get('cv_folds', 5),
            'cv_type': params.get('cv_type', 'auto')
        }
        
        cv_info = f" with {cv_config['cv_folds']}-fold CV" if cv_config['use_cv'] else ""
        print(f"🎯 Training {model} on {dataset}{cv_info}")
        
        # Update session data
        self.session_data['models_used'].add(model)
        self.session_data['datasets_used'].add(dataset)
        self.session_data['last_model'] = model
        self.session_data['last_dataset'] = dataset
        self.session_data['command_count'] += 1
        
        try:
            train_from_prompt(prompt, cv_config=cv_config)
            print(f"✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Check if model and dataset names are correct")

    def handle_tune_command(self, params: Dict):
        """Handle hyperparameter tuning with enhanced validation"""
        params = self.validate_and_enhance_params('tune', params)
        if not params:
            return
        
        model = params.get('model')
        dataset = params.get('dataset')
        n_trials = params.get('n_trials', 50)
        
        prompt = f"model: {model} dataset: {dataset} n_trials: {n_trials}"
        print(f"🔬 Optimizing {model} hyperparameters on {dataset} ({n_trials} trials)")
        
        # Update session data
        self.session_data['models_used'].add(model)
        self.session_data['datasets_used'].add(dataset)
        self.session_data['last_model'] = model
        self.session_data['last_dataset'] = dataset
        self.session_data['command_count'] += 1
        
        try:
            tune_from_prompt(prompt)
            print(f"✅ Hyperparameter tuning completed!")
        except Exception as e:
            print(f"❌ Tuning failed: {e}")

    def handle_compare_command(self, params: Dict):
        """Handle model comparison with enhanced validation"""
        params = self.validate_and_enhance_params('compare', params)
        if not params:
            return
        
        models = params.get('models', [])
        dataset = params.get('dataset')
        
        if not models:
            # Use default models for comparison
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
            print(f"🏁 Using default models: {', '.join(models)}")
        
        print(f"🏁 Comparing {len(models)} models on {dataset}")
        
        try:
            from cli import compare
            compare(
                prompt=f"dataset: {dataset}",
                models=','.join(models),
                save_results=True,
                use_cv=params.get('use_cv', False),
                cv_folds=params.get('cv_folds', 5),
                generate_viz=True
            )
            
            # Update session data
            self.session_data['models_used'].update(models)
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
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

    def handle_advisor_command(self, params: Dict):
        """Handle model advisor with enhanced validation"""
        params = self.validate_and_enhance_params('advisor', params)
        if not params:
            return

        
        dataset = params.get('dataset')
        print(f"🧠 Getting intelligent recommendations for {dataset}")
        
        try:
            from cli import advisor
            advisor(
                dataset=dataset,
                detailed=params.get('detailed', True),
                auto_compare=params.get('auto_compare', False),
                save_report=True
            )
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
        except Exception as e:
            print(f"❌ Advisor failed: {e}")
            print("💡 Make sure the dataset name is correct")

    def handle_engineer_command(self, params: Dict):
        """Handle feature engineering with enhanced validation"""
        params = self.validate_and_enhance_params('engineer', params)
        if not params:
            return
        
        dataset = params.get('dataset')
        strategy = params.get('strategy', 'comprehensive')
        
        print(f"🔧 Engineering features for {dataset} using {strategy} strategy")
        
        try:
            from cli import engineer
            prompt = f"engineer features for {dataset}"
            if strategy != 'comprehensive':
                prompt += f" strategy {strategy}"
            
            engineer(
                prompt=prompt,
                dataset=dataset,
                output=f"engineered_{dataset}.csv",
                save_importance=True,
                show_report=True
            )
            
            elif action == 'list':
                self.handle_list_command(params.get('args', []))
            
            elif action == 'train':
                self.handle_train_command(params)
            
            elif action == 'tune':
                self.handle_tune_command(params)
            
            elif action == 'compare':
                self.handle_compare_command(params)
            
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

            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1

            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")


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

    def handle_select_command(self, params: Dict):
        """Handle feature selection with enhanced validation"""
        params = self.validate_and_enhance_params('select', params)
        if not params:
            return

        
        dataset = params.get('dataset')
        method = params.get('method', 'univariate')
        k = params.get('k', 50)
        
        print(f"🎯 Selecting top {k} features from {dataset} using {method}")
        
        try:
            from cli import select
            select(
                dataset=dataset,
                method=method,
                k=k,
                output=f"selected_{dataset}.csv",
                show_importance=True
            )
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            
        except Exception as e:
            print(f"❌ Feature selection failed: {e}")

    def handle_list_command(self, args: List[str]):
        """Handle list commands with better formatting"""
        if not args:

            print("📋 Available list options: models, datasets, frameworks, history, deployments")

            return
        
        command = args[0].lower()
        
        if command == 'models':
            self.list_models()
        elif command == 'datasets':
            self.list_datasets()
        elif command == 'frameworks':
            self.list_frameworks()
        elif command == 'history':
            self.show_history()
        elif command == 'deployments':
            self.list_deployments()
        else:
            print(f"❓ Unknown list command: {command}")

    def list_models(self):
        """List available models with better formatting"""
        print("📊 Available Models:")
        
        frameworks = {}
        for model_name in get_available_models():
            fw = MODEL_REGISTRY[model_name]["framework"]
            if fw not in frameworks:
                frameworks[fw] = []
            frameworks[fw].append(model_name)
        
        for fw, models in frameworks.items():
            print(f"\n🔧 {fw.upper()}:")
            # Display in columns
            for i, model in enumerate(models):
                if i % 2 == 0:
                    print(f"  • {model:<25}", end="")
                else:
                    print(f"• {model}")
            if len(models) % 2 == 1:
                print()

    def list_datasets(self):
        """List known datasets with better formatting"""
        print("🗂️  Known Datasets:")
        datasets = sorted(list(self.known_datasets))
        
        # Display in 4 columns
        cols = 4
        for i in range(0, len(datasets), cols):
            row = datasets[i:i+cols]
            print("  " + "".join(f"• {ds:<18}" for ds in row))

    def list_frameworks(self):
        """List available frameworks"""
        frameworks = set(config['framework'] for config in MODEL_REGISTRY.values())
        print("🔧 Available Frameworks:")
        for fw in sorted(frameworks):
            print(f"  • {fw}")

    def list_deployments(self):
        """List deployed models from session"""
        if not self.session_data['deployed_models']:
            print("📦 No models deployed in this session")
            return
        
        print("📦 Deployed Models (This Session):")
        for i, deployment in enumerate(self.session_data['deployed_models'], 1):
            print(f"  {i}. {deployment['model_file']}")
            print(f"     Output: {deployment['output_dir']}")
            print(f"     Port: {deployment['port']}")
            print(f"     Time: {deployment['timestamp']}")
            print()

    def show_history(self):
        """Show command history with better formatting"""
        try:
            history_length = readline.get_current_history_length()
            if history_length == 0:
                print("📜 No command history available")
                return
            
            print("📜 Recent Commands:")
            start = max(1, history_length - 9)
            for i in range(start, history_length + 1):
                try:
                    cmd = readline.get_history_item(i)
                    if cmd:
                        print(f"  {i:2d}: {cmd}")
                except Exception:
                    pass
        except NameError:
            print("📜 Command history not available")

    def show_session_stats(self):
        """Show enhanced session statistics"""
        if self.session_data['command_count'] > 0:
            print(f"\n📊 Session Summary:")
            print(f"  Commands executed: {self.session_data['command_count']}")
            
            if self.session_data['models_used']:
                models_list = ', '.join(sorted(self.session_data['models_used']))
                print(f"  Models used: {models_list}")
            
            if self.session_data['datasets_used']:

                print(f"  Datasets used: {', '.join(sorted(self.session_data['datasets_used']))}")
            if self.session_data['deployed_models']:
                print(f"  Models deployed: {len(self.session_data['deployed_models'])}")

                datasets_list = ', '.join(sorted(self.session_data['datasets_used']))
                print(f"  Datasets explored: {datasets_list}")
            

            if self.session_data['last_model']:
                print(f"  Last model: {self.session_data['last_model']}")
            
            if self.session_data['last_dataset']:
                print(f"  Last dataset: {self.session_data['last_dataset']}")

    def run(self):
        """Main interactive loop with enhanced UX"""
        self.print_banner()
        
        try:
            while True:
                try:
                    command = input("epoch> ").strip()
                    
                    if not command:
                        continue
                    
                    # Parse and execute with enhanced understanding
                    action, params = self.enhanced_command_parsing(command)
                    
                    if not self.execute_command(action, params):
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Use 'exit' to quit gracefully")
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
        
        finally:
            self.show_session_stats()
            print("\n🎉 Thanks for using Epoch CLI!")
            print("💡 Your command history has been saved for next time")


def main():
    """Entry point for enhanced interactive CLI"""
    cli = EpochInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
