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
    """Complete Interactive command-line interface for Epoch ML toolkit"""
    
    def __init__(self):
        self.version = "2.1.0"
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
        
        # FIXED: Comprehensive command patterns with better regex
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
            'advisor': [
                r'(?:analyze|advisor?|recommend|suggest)\s+(?:models?\s+(?:for\s+)?)?(.+?)(?:\s+dataset)?(?:\s+(?:with|using)\s+(.+))?$',
                r'what\s+(?:models?|algorithms?)\s+(?:should\s+i\s+use\s+(?:for\s+)?|work\s+best\s+(?:for\s+)?)?(.+?)(?:\s+dataset)?(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'engineer': [
                # FIXED: More specific and working patterns
                r'(?:engineer|create|generate|build)\s+(?:polynomial\s+)?features?\s+(?:for\s+)?(\w+)(?:\s+(?:with|using)\s+(.+))?$',
                r'(?:create|build)\s+(?:polynomial|mathematical|statistical)\s+features?\s+(?:for\s+)?(\w+)(?:\s+(?:with|using)\s+(.+))?$'
            ],
            'select': [
                # FIXED: Better capture groups and patterns
                r'select\s+(?:(?:top|best)\s+)?(?:(\d+)\s+)?features?\s+(?:from\s+)?(\w+)(?:\s+(?:using|with|method)\s+(\w+))?$',
                r'choose\s+(?:(?:best|top)\s+)?(?:(\d+)\s+)?features?\s+(?:from\s+)?(\w+)(?:\s+(?:using|with|method)\s+(\w+))?$',
                r'pick\s+(?:(?:top|best)\s+)?(?:(\d+)\s+)?features?\s+(?:from\s+)?(\w+)(?:\s+(?:using|with|method)\s+(\w+))?$'
            ],
            'deploy': [
                r'deploy\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'(?:create|generate)\s+(?:fastapi\s+)?(?:api|deployment)\s+(?:for\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$',
                r'serve\s+(?:model\s+)?(.+?)(?:\s+(?:to|as)\s+(.+?))?(?:\s+(?:on\s+port\s+|port\s+)(\d+))?$'
            ]
        }
        
        # Available datasets (comprehensive list)
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
            'commands': ['train', 'tune', 'compare', 'benchmark', 'advisor', 'analyze', 'engineer', 'select', 'deploy', 'list', 'help', 'history', 'clear', 'exit']
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
        """Enhanced tab completion with context awareness"""
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
                
                if any(word in ['on', 'dataset', 'data', 'from', 'for', 'using'] for word in words[-3:]):
                    matches.extend([dataset for dataset in self.suggestions['datasets'] if dataset.startswith(text)])
                
                if any(word in ['analyze', 'advisor', 'recommend', 'suggest'] for word in words):
                    matches.extend([dataset for dataset in self.suggestions['datasets'] if dataset.startswith(text)])
                
                if any(word in ['compare', 'benchmark'] for word in words):
                    matches.extend([model for model in self.suggestions['models'] if model.startswith(text)])
                    common_combos = [
                        'random_forest,xgboost,lightgbm',
                        'sklearn models',
                        'tree models',
                        'ensemble models'
                    ]
                    matches.extend([combo for combo in common_combos if combo.startswith(text)])
                
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
        print("🤖 Complete Epoch CLI v{} - Full-Featured ML Command Shell".format(self.version))
        print("="*70)
        print("🎯 Train, tune, compare, analyze, and deploy ML models")
        print("🔧 Feature engineering and selection capabilities")
        print("🚀 FastAPI deployment generation")
        print("📚 Type 'help' for commands or try these examples:")
        print("   • train random forest on penguins")
        print("   • engineer features for titanic")
        print("   • select top 20 features from wine")
        print("   • deploy my_model.joblib")
        print("   • analyze iris dataset")
        print("❌ Type 'exit' or 'quit' to leave")
        print("="*70 + "\n")

    def print_help(self):
        """Print comprehensive help information"""
        help_text = """
🔧 COMPLETE EPOCH CLI COMMANDS

📊 MODEL TRAINING & OPTIMIZATION:
  train <model> on <dataset> [options]      Train a model
  tune <model> on <dataset> [options]       Optimize hyperparameters  
  compare <models> on <dataset> [options]   Compare multiple models
  benchmark <dataset> [options]             Comprehensive benchmarking

🧠 INTELLIGENT ANALYSIS:
  analyze <dataset>                         Get intelligent model recommendations
  advisor <dataset>                         Detailed model suggestions with insights
  recommend models for <dataset>            Quick model recommendations

🔧 FEATURE ENGINEERING & SELECTION:
  engineer features for <dataset>           Comprehensive feature engineering
  generate polynomial features for <dataset>  Generate polynomial/interaction features
  select features from <dataset>            Intelligent feature selection
  select top N features from <dataset>      Select specific number of features

🚀 MODEL DEPLOYMENT:
  deploy <model_file>                       Generate FastAPI deployment
  deploy <model_file> port <port>           Deploy with custom port
  serve model <model_file>                  Create API service

🛠️  UTILITY COMMANDS:
  list models [framework]                   Show available models
  list datasets                            Show known datasets  
  list frameworks                          Show available frameworks
  help                                     Show this help
  history                                  Command history
  clear                                    Clear screen
  exit / quit                              Exit CLI

🎯 NATURAL LANGUAGE EXAMPLES:
  ✅ "train random forest on penguins with cv"
  ✅ "tune xgboost on titanic with 100 trials"  
  ✅ "compare random_forest,xgboost,svm on wine"
  ✅ "select top 20 features from diabetes using recursive"
  ✅ "engineer features for breast_cancer"
  ✅ "analyze penguins dataset"
  ✅ "deploy outputs/best_model.joblib port 8080"

⚙️  ADVANCED OPTIONS:
  • with cv / with cross-validation         Enable cross-validation
  • with N-fold cv                          Specify CV folds  
  • with N trials                           Set HPO trials
  • framework sklearn/xgboost/lightgbm      Filter by framework
  • using <method>                          Specify algorithm/method
  • port <number>                           Custom deployment port

📁 Available Models: {}
🗂️  Known Datasets: {}

💡 TIP: The CLI understands natural language - be flexible with your commands!
        """.format(
            ', '.join(sorted(self.suggestions['models'])[:10]) + '...',
            ', '.join(sorted(self.suggestions['datasets'])[:10]) + '...'
        )
        print(help_text)

    def parse_cli_flags(self, command: str) -> Tuple[str, Dict]:
        """Parse CLI-style flag commands for deployment"""
        try:
            parts = shlex.split(command)
            if not parts:
                return 'unknown', {}
            
            cmd = parts[0].lower()
            
            if cmd == 'deploy':
                params = {'action': 'deploy'}
                i = 1
                
                while i < len(parts):
                    if parts[i].startswith('--'):
                        flag = parts[i][2:]
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
                        if 'model_file' not in params:
                            params['model_file'] = parts[i]
                        i += 1
                
                return 'deploy', params
        
        except Exception as e:
            print(f"⚠️  Error parsing CLI flags: {e}")
        
        return 'unknown', {}

    def enhanced_dataset_extraction(self, command: str, context_parts: List[str] = None) -> Optional[str]:
        """Enhanced dataset name extraction with multiple strategies"""
        command_lower = command.lower()
        
        # Strategy 1: Look for explicit dataset mentions
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
                skip_words = {
                    'features', 'feature', 'models', 'model', 'top', 'best', 'using', 'with', 
                    'method', 'select', 'create', 'train', 'tune', 'compare', 'analyze',
                    'polynomial', 'mathematical', 'statistical', 'data', 'dataset',
                    'from', 'for', 'on', 'the', 'and', 'or', 'is', 'are', 'port'
                }
                
                if candidate and candidate not in skip_words:
                    matched_dataset = self.fuzzy_match_dataset(candidate)
                    if matched_dataset:
                        return matched_dataset
                    elif len(candidate) > 2 and candidate.isalpha():
                        return candidate
        
        # Strategy 2: Check context parts
        if context_parts:
            for part in context_parts:
                if part:
                    matched = self.fuzzy_match_dataset(part)
                    if matched:
                        return matched
        
        # Strategy 3: Look for dataset names anywhere
        words = command_lower.split()
        for word in words:
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
        
        # Partial matching
        for dataset in self.known_datasets:
            if candidate in dataset or dataset in candidate:
                min_len = min(len(candidate), len(dataset))
                max_len = max(len(candidate), len(dataset))
                similarity = min_len / max_len
                
                if similarity >= 0.6:
                    return dataset
        
        # Common aliases
        aliases = {
            'breast': 'breast_cancer',
            'cancer': 'breast_cancer', 
            'penguin': 'penguins',
            'housing': 'california_housing',
            'tip': 'tips',
            'car': 'cars'
        }
        
        for alias, dataset in aliases.items():
            if alias in candidate or candidate in alias:
                return dataset
        
        # Return original if it looks valid
        if candidate.isalpha() and len(candidate) > 2:
            return candidate
        
        return None

    def parse_natural_language(self, command: str) -> Tuple[str, Dict]:
        """Enhanced natural language parsing"""
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
        
        # Try CLI-style parsing for deploy commands
        if command.startswith('deploy') and ('--' in command or '-' in command):
            return self.parse_cli_flags(command)
        
        # Parse complex commands using patterns
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if action == 'advisor':
                        dataset_part = groups[0] if groups[0] else ""
                        options_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        params = self.extract_advisor_parameters(command, dataset_part, options_part)
                        params['action'] = action
                        return action, params
                    
                    elif action == 'select':
                        # FIXED: Better parameter handling
                        k_part = None
                        dataset_part = None
                        method_part = None
                        
                        # Handle different group configurations
                        if len(groups) >= 3:
                            k_part = groups[0] if groups[0] and groups[0].isdigit() else None
                            dataset_part = groups[1] if groups[1] else groups[0] if not k_part else ""
                            method_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        elif len(groups) == 2:
                            if groups[0] and groups[0].isdigit():
                                k_part = groups[0]
                                dataset_part = groups[1]
                            else:
                                dataset_part = groups[0]
                                method_part = groups[1]
                        elif len(groups) == 1:
                            dataset_part = groups[0]
                        
                        params = self.extract_select_parameters(command, k_part, dataset_part, method_part)
                        params['action'] = action
                        return action, params
                    
                    elif action == 'engineer':
                        dataset_part = groups[0] if groups[0] else ""
                        options_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        
                        params = self.extract_engineer_parameters(command, dataset_part, options_part)
                        params['action'] = action
                        return action, params
                    
                    elif action == 'deploy':
                        model_part = groups[0] if groups[0] else ""
                        output_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        port_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        
                        params = self.extract_deploy_parameters(command, model_part, output_part, port_part)
                        params['action'] = action
                        return action, params
                    
                    else:
                        # Standard parameter extraction for train/tune/compare
                        model_part = groups[0] if groups[0] else ""
                        dataset_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        options_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        
                        params = self.extract_parameters(command, model_part, dataset_part, options_part)
                        params['action'] = action
                        return action, params
        
        return 'unknown', {'original_command': command}

    def extract_advisor_parameters(self, full_command: str, dataset_part: str, options_part: str) -> Dict:
        """Extract parameters for advisor command"""
        params = {}
        
        if dataset_part:
            dataset_part = dataset_part.strip()
            dataset_part = re.sub(r'\b(the|dataset|data|models?|for)\b', '', dataset_part, flags=re.IGNORECASE).strip()
            dataset_part = re.sub(r'\s+', ' ', dataset_part).strip()
            
            matched_dataset = self.enhanced_dataset_extraction(dataset_part)
            if matched_dataset:
                params['dataset'] = matched_dataset
            else:
                params['dataset'] = dataset_part
        
        # Extract options
        if re.search(r'\b(?:detailed|full|complete)\b', full_command, re.IGNORECASE):
            params['detailed'] = True
        elif re.search(r'\b(?:summary|brief|quick)\b', full_command, re.IGNORECASE):
            params['detailed'] = False
        else:
            params['detailed'] = True
        
        if re.search(r'\b(?:auto.?compare|compare.?auto)\b', full_command, re.IGNORECASE):
            params['auto_compare'] = True
        
        if re.search(r'\b(?:interpretable|explainable|transparent)\b', full_command, re.IGNORECASE):
            params['prefer_interpretable'] = True
        
        if re.search(r'\b(?:fast|quick|speed)\b', full_command, re.IGNORECASE):
            params['prefer_fast'] = True
        
        return params

    def extract_select_parameters(self, full_command: str, k_part: str, dataset_part: str, method_part: str) -> Dict:
        """FIXED: Extract parameters for feature selection command"""
        params = {}
        
        # Extract number of features - FIXED
        if k_part and k_part.isdigit():
            params['k'] = int(k_part)
        else:
            # Look for number in the command more broadly
            k_matches = re.findall(r'\b(\d+)\b', full_command)
            if k_matches:
                # Take the first reasonable number (not too large, not 1)
                for num in k_matches:
                    if 2 <= int(num) <= 1000:
                        params['k'] = int(num)
                        break
            else:
                params['k'] = 50  # Default
        
        # Extract dataset - FIXED with better logic
        dataset = None
        if dataset_part:
            # Clean the dataset part
            cleaned = re.sub(r'\b(from|features?|feature|dataset|data|top|best|select|choose|pick)\b', '', dataset_part, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                dataset = self.fuzzy_match_dataset(cleaned)
                if not dataset and len(cleaned) > 2 and cleaned.isalpha():
                    dataset = cleaned
        
        # If no dataset found, try broader extraction
        if not dataset:
            # Look for "from <dataset>" pattern
            from_match = re.search(r'from\s+(\w+)', full_command, re.IGNORECASE)
            if from_match:
                candidate = from_match.group(1)
                if candidate.lower() not in ['features', 'feature', 'dataset', 'data', 'top', 'best']:
                    dataset = self.fuzzy_match_dataset(candidate)
                    if not dataset and len(candidate) > 2:
                        dataset = candidate
        
        if dataset:
            params['dataset'] = dataset
        
        # Extract selection method - FIXED
        method = 'univariate'  # Default
        if method_part:
            method_part = method_part.lower().strip()
            methods = ['univariate', 'recursive', 'model_based', 'correlation', 'variance']
            for m in methods:
                if m in method_part or m.replace('_', ' ') in method_part:
                    method = m
                    break
        else:
            # Check full command for method keywords
            if 'recursive' in full_command.lower():
                method = 'recursive'
            elif 'model' in full_command.lower() and ('based' in full_command.lower() or 'base' in full_command.lower()):
                method = 'model_based'
            elif 'correlation' in full_command.lower():
                method = 'correlation'
            elif 'variance' in full_command.lower():
                method = 'variance'
        
        params['method'] = method
        return params

    def extract_engineer_parameters(self, full_command: str, dataset_part: str, options_part: str) -> Dict:
        """FIXED: Extract parameters for feature engineering command"""
        params = {}
        
        # Extract dataset - FIXED with better logic
        dataset = None
        if dataset_part:
            # Clean the dataset part
            cleaned = re.sub(r'\b(for|features?|feature|engineer|create|generate|build|polynomial)\b', '', dataset_part, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                dataset = self.fuzzy_match_dataset(cleaned)
                if not dataset and len(cleaned) > 2 and cleaned.isalpha():
                    dataset = cleaned
        
        # If no dataset found, try broader extraction
        if not dataset:
            # Look for "for <dataset>" pattern
            for_match = re.search(r'for\s+(\w+)', full_command, re.IGNORECASE)
            if for_match:
                candidate = for_match.group(1)
                if candidate.lower() not in ['features', 'feature']:
                    dataset = self.fuzzy_match_dataset(candidate)
                    if not dataset and len(candidate) > 2:
                        dataset = candidate
        
        if dataset:
            params['dataset'] = dataset
        
        # Extract strategy based on keywords in full command
        if 'polynomial' in full_command.lower():
            params['strategy'] = 'polynomial'
        elif 'mathematical' in full_command.lower():
            params['strategy'] = 'mathematical'  
        elif 'statistical' in full_command.lower():
            params['strategy'] = 'statistical'
        else:
            params['strategy'] = 'comprehensive'
        
        return params

    def extract_deploy_parameters(self, full_command: str, model_part: str, output_part: str, port_part: str) -> Dict:
        """Extract parameters for deployment command"""
        params = {}
        
        # Extract model file
        if model_part:
            params['model_file'] = model_part.strip()
        
        # Extract output directory
        if output_part:
            params['output_dir'] = output_part.strip()
        else:
            params['output_dir'] = 'deployment_fastapi'
        
        # Extract port
        if port_part and port_part.isdigit():
            params['port'] = int(port_part)
        else:
            # Look for port in full command
            port_match = re.search(r'(?:port\s+|:)(\d+)', full_command)
            if port_match:
                params['port'] = int(port_match.group(1))
            else:
                params['port'] = 8000
        
        return params

    def extract_parameters(self, full_command: str, model_part: str, dataset_part: str, options_part: str) -> Dict:
        """Extract parameters for standard commands (train/tune/compare)"""
        params = {}
        
        # Extract models
        if model_part:
            model_part = model_part.strip()
            
            if ',' in model_part:
                # Multiple models
                model_candidates = [m.strip() for m in model_part.split(',')]
                valid_models = []
                available_models = get_available_models()
                
                for candidate in model_candidates:
                    clean_candidate = re.sub(r'\b(a|an|the|model|classifier|regressor)\b', '', candidate, flags=re.IGNORECASE).strip()
                    clean_candidate = re.sub(r'\s+', '_', clean_candidate)
                    
                    matched = self.fuzzy_match_model(clean_candidate, available_models)
                    if matched:
                        valid_models.append(matched)
                    else:
                        valid_models.append(clean_candidate)
                
                params['models'] = valid_models
            else:
                # Single model
                model_part = re.sub(r'\b(a|an|the|model|classifier|regressor)\b', '', model_part, flags=re.IGNORECASE).strip()
                model_part = re.sub(r'\s+', '_', model_part)
                
                available_models = get_available_models()
                matched = self.fuzzy_match_model(model_part, available_models)
                if matched:
                    params['model'] = matched
                else:
                    params['model'] = model_part
        
        # Handle framework specification
        if model_part and ('models' in model_part or 'framework' in full_command):
            framework_match = re.search(r'(sklearn|xgboost|lightgbm|tensorflow|pytorch|catboost)\s+models?', model_part, re.IGNORECASE)
            if framework_match:
                params['framework'] = framework_match.group(1).lower()
                if 'model' in params:
                    del params['model']
                if 'models' in params:
                    del params['models']
        
        # Extract dataset
        if dataset_part:
            matched_dataset = self.enhanced_dataset_extraction(dataset_part)
            if matched_dataset:
                params['dataset'] = matched_dataset
        
        # Extract CV settings
        cv_match = re.search(r'(\d+)[-\s]*fold\s+(?:cv|cross.?validation)', full_command, re.IGNORECASE)
        if cv_match:
            params['cv_folds'] = int(cv_match.group(1))
            params['use_cv'] = True
        elif re.search(r'\b(?:cv|cross.?validation)\b', full_command, re.IGNORECASE):
            params['use_cv'] = True
            params['cv_folds'] = 5
        
        # Extract CV strategy
        if re.search(r'stratified', full_command, re.IGNORECASE):
            params['cv_type'] = 'stratified'
        elif re.search(r'kfold|k.fold', full_command, re.IGNORECASE):
            params['cv_type'] = 'kfold'
        
        # Extract trials for tuning
        trials_match = re.search(r'(\d+)\s+trials?', full_command, re.IGNORECASE)
        if trials_match:
            params['n_trials'] = int(trials_match.group(1))
        
        # Extract framework filter
        framework_match = re.search(r'framework\s+(\w+)', full_command, re.IGNORECASE)
        if framework_match:
            params['framework'] = framework_match.group(1)
        
        return params

    def fuzzy_match_model(self, candidate: str, available_models: List[str]) -> Optional[str]:
        """Fuzzy match model names"""
        if not candidate:
            return None
        
        candidate = candidate.lower().strip()
        candidate = re.sub(r'\s+', '_', candidate)
        
        # Exact match
        if candidate in available_models:
            return candidate
        
        # Partial matching
        best_match = None
        best_score = 0
        
        for model in available_models:
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
        
        return best_match if best_score > 0.4 else None

    def prompt_for_model_file(self) -> Optional[str]:
        """Interactive model file selection"""
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
                    custom_path = input("Enter model file path: ").strip()
                    return custom_path if custom_path else None
        except (ValueError, KeyboardInterrupt):
            pass
        
        return None

    def validate_and_enhance_params(self, action: str, params: Dict) -> Dict:
        """Comprehensive parameter validation and enhancement"""
        enhanced_params = params.copy()

        # Actions that need datasets
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

        # Actions that need models
        if action in ['train', 'tune']:
            if not enhanced_params.get('model'):
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

        # Deploy-specific logic
        if action == 'deploy':
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
            model_file = model_file.strip('"\'')
            
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
                    for i, f in enumerate(model_files[:10], 1):
                        print(f"  {i}. {f}")
                    if len(model_files) > 10:
                        print(f"  ... and {len(model_files) - 10} more")
                else:
                    print("  No model files found in current directory, outputs/, or outputs/models/")
                return {}

        return enhanced_params

    def handle_train_command(self, params: Dict):
        """Handle training commands with comprehensive validation"""
        params = self.validate_and_enhance_params('train', params)
        if not params:
            return
            
        model = params.get('model')
        dataset = params.get('dataset')
        
        prompt = f"model: {model} dataset: {dataset}"
        
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
        
        try:
            train_from_prompt(prompt, cv_config=cv_config)
            print(f"✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Check if the model and dataset names are correct")

    def handle_tune_command(self, params: Dict):
        """Handle hyperparameter tuning with comprehensive validation"""
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
            print("💡 Check if the model and dataset names are correct")

    def handle_compare_command(self, params: Dict):
        """Handle model comparison with comprehensive validation"""
        params = self.validate_and_enhance_params('compare', params)
        if not params:
            return
            
        models = params.get('models', [])
        dataset = params.get('dataset')
        framework = params.get('framework')
        
        if not models and not framework:
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
            print(f"🏁 Using default models for comparison: {', '.join(models)}")
        
        try:
            from cli import compare
            
            prompt = f"dataset: {dataset}"
            
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
            print(f"✅ Model comparison completed!")
            
        except ImportError:
            print("❌ Compare function not available - check if cli.py exists")
        except Exception as e:
            print(f"❌ Comparison failed: {e}")

    def handle_advisor_command(self, params: Dict):
        """Handle model advisor with comprehensive validation"""
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
                save_report=True,
                prefer_interpretable=params.get('prefer_interpretable', False),
                prefer_fast=params.get('prefer_fast', False)
            )
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            print(f"✅ Model advisor analysis completed!")
            
        except ImportError:
            print("❌ Advisor function not available - check if cli.py exists")
        except Exception as e:
            print(f"❌ Advisor failed: {e}")
            print("💡 Make sure the dataset name is correct")

    def handle_engineer_command(self, params: Dict):
        """FIXED: Handle feature engineering with proper error handling"""
        params = self.validate_and_enhance_params('engineer', params)
        if not params:
            return
            
        dataset = params.get('dataset')
        strategy = params.get('strategy', 'comprehensive')
        
        print(f"🔧 Engineering features for {dataset} using {strategy} strategy")
        
        try:
            # Try importing the engineer function
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
            except ImportError:
                # If cli.py doesn't exist, create a simple implementation
                print("⚠️  CLI engineer function not found. Using basic implementation...")
                self.basic_feature_engineering(dataset, strategy)
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            print(f"✅ Feature engineering completed!")
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            print(f"💡 Attempted: dataset={dataset}, strategy={strategy}")

    def handle_select_command(self, params: Dict):
        """FIXED: Handle feature selection with proper error handling"""
        params = self.validate_and_enhance_params('select', params)
        if not params:
            return
            
        dataset = params.get('dataset')
        method = params.get('method', 'univariate')
        k = params.get('k', 50)
        
        print(f"🎯 Selecting top {k} features from {dataset} using {method}")
        
        try:
            # Try importing the select function
            try:
                from cli import select
                select(
                    dataset=dataset,
                    method=method,
                    k=k,
                    output=f"selected_{dataset}.csv",
                    show_importance=True
                )
            except ImportError:
                # If cli.py doesn't exist, create a simple implementation
                print("⚠️  CLI select function not found. Using basic implementation...")
                self.basic_feature_selection(dataset, method, k)
            
            self.session_data['datasets_used'].add(dataset)
            self.session_data['last_dataset'] = dataset
            self.session_data['command_count'] += 1
            print(f"✅ Feature selection completed!")
            
        except Exception as e:
            print(f"❌ Feature selection failed: {e}")
            print(f"💡 Attempted: dataset={dataset}, method={method}, k={k}")

    def basic_feature_selection(self, dataset: str, method: str, k: int):
        """Basic feature selection implementation as fallback"""
        print(f"📊 Basic feature selection: {method} method, top {k} features from {dataset}")
        print("💡 This is a placeholder - implement actual feature selection logic")
        
        # Here you could add basic feature selection logic using sklearn
        # For now, just showing what was attempted
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Features to select: {k}")

    def basic_feature_engineering(self, dataset: str, strategy: str):
        """Basic feature engineering implementation as fallback"""
        print(f"🔧 Basic feature engineering: {strategy} strategy for {dataset}")
        print("💡 This is a placeholder - implement actual feature engineering logic")
        
        # Here you could add basic feature engineering logic
        # For now, just showing what was attempted
        print(f"   Dataset: {dataset}")
        print(f"   Strategy: {strategy}")

    def handle_deploy_command(self, params: Dict):
        """Handle deployment commands with comprehensive validation"""
        params = self.validate_and_enhance_params('deploy', params)
        if not params:
            return
            
        model_file = params.get('model_file')
        output_dir = params.get('output_dir', 'deployment_fastapi')
        port = params.get('port', 8000)
        
        print(f"🚀 Deploying model: {model_file}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🌐 Port: {port}")
        
        try:
            sys.path.append('core')
            from core.fastapi_deployer import generate_fastapi_app
            
            generate_fastapi_app(model_file, output_dir, port)
            
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
        """Interactive model selection with enhanced UX"""
        available_models = get_available_models()
        frameworks = {}
        
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
        """Interactive dataset selection with enhanced UX"""
        print("\n🗂️  Available datasets:")
        datasets = sorted(list(self.known_datasets))
        
        # Display in columns for better readability
        cols = 3
        for i in range(0, len(datasets), cols):
            row = datasets[i:i+cols]
            print("  " + "".join(f"{j+i+1:2d}. {ds:<20}" for j, ds in enumerate(row)))
        
        print(f"\n  {len(datasets) + 1:2d}. Enter custom dataset name")
        print(f"  {len(datasets) + 2:2d}. Skip (cancel)")
        
        try:
            choice = input(f"\nSelect a dataset (1-{len(datasets) + 2}): ").strip()
            
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

    def execute_command(self, action: str, params: Dict) -> bool:
        """Execute the parsed command with comprehensive error handling"""
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
            
            elif action == 'deploy':
                self.handle_deploy_command(params)

            elif action == 'unknown':
                original_cmd = params.get('original_command', '')
                print(f"❓ Couldn't understand command: '{original_cmd}'")
                
                suggestions = self.suggest_corrections(original_cmd)
                if suggestions:
                    print("💡 Did you mean:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
                    
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
            import traceback
            traceback.print_exc()
            return True

    def suggest_corrections(self, command: str) -> List[str]:
        """Generate smart suggestions for unknown commands"""
        suggestions = []
        command_lower = command.lower()
        
        # Common typo corrections
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
            'delpoy': 'deploy', 'depooy': 'deploy', 'depliy': 'deploy',
            'enginer': 'engineer', 'selct': 'select'
        }
        
        for typo, correction in typo_corrections.items():
            if typo in command_lower:
                corrected = command_lower.replace(typo, correction)
                suggestions.append(corrected)
        
        # Template-based suggestions
        if any(word in command_lower for word in ['train', 'fit', 'model']):
            suggestions.extend([
                "train random_forest on iris",
                "train xgboost on penguins with cv"
            ])
        
        if any(word in command_lower for word in ['select', 'feature']):
            suggestions.extend([
                "select features from penguins dataset",
                "select top 20 features from wine using recursive"
            ])
        
        if any(word in command_lower for word in ['engineer', 'create']):
            suggestions.extend([
                "engineer features for titanic",
                "create polynomial features for diabetes"
            ])
        
        if any(word in command_lower for word in ['deploy', 'api']):
            suggestions.extend([
                "deploy model.joblib",
                "deploy outputs/best_model.pkl port 8080"
            ])
        
        return suggestions[:5]

    def handle_list_command(self, args: List[str]):
        """Handle list commands with enhanced formatting"""
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
        """List available models with enhanced formatting"""
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
            
            frameworks = {}
            for model_name in get_available_models():
                fw = MODEL_REGISTRY[model_name]["framework"]
                if fw not in frameworks:
                    frameworks[fw] = []
                frameworks[fw].append(model_name)
            
            for fw, models in frameworks.items():
                print(f"\n🔧 {fw.upper()}:")
                for i, model in enumerate(models):
                    if i % 2 == 0:
                        print(f"  • {model:<25}", end="")
                    else:
                        print(f"• {model}")
                if len(models) % 2 == 1:
                    print()

    def list_datasets(self):
        """List known datasets with enhanced formatting"""
        print("🗂️  Known Datasets:")
        datasets = sorted(list(self.known_datasets))
        
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

    def show_history(self):
        """Show recent command history with enhanced formatting"""
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
            print("📜 Command history not available (readline not loaded)")

    def show_session_stats(self):
        """Show comprehensive session statistics"""
        if self.session_data['command_count'] > 0:
            print(f"\n📊 Session Summary:")
            print(f"  Commands executed: {self.session_data['command_count']}")
            if self.session_data['models_used']:
                print(f"  Models used: {', '.join(sorted(self.session_data['models_used']))}")
            if self.session_data['datasets_used']:
                print(f"  Datasets used: {', '.join(sorted(self.session_data['datasets_used']))}")
            if self.session_data['last_model']:
                print(f"  Last model: {self.session_data['last_model']}")
            if self.session_data['last_dataset']:
                print(f"  Last dataset: {self.session_data['last_dataset']}")
            if self.session_data['deployed_models']:
                print(f"  Models deployed: {len(self.session_data['deployed_models'])}")
                for deployment in self.session_data['deployed_models'][-3:]:  # Show last 3
                    print(f"    • {deployment['model_file']} -> {deployment['output_dir']} ({deployment['timestamp']})")

    def run(self):
        """Main interactive loop with comprehensive UX"""
        self.print_banner()
        
        try:
            while True:
                try:
                    command = input("epoch> ").strip()
                    
                    if not command:
                        continue
                    
                    # Parse and execute command
                    action, params = self.parse_natural_language(command)
                    
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
    """Entry point for the complete interactive CLI"""
    cli = EpochInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()