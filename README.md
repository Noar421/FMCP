# MCP Arduino Relay Controller

Application Python pour contrôler des relais Arduino via le protocole MCP (Model Context Protocol) avec un LLM local (Llama 3.2).

## 🏗️ Architecture

```
┌─────────────────┐      HTTP/MCP       ┌─────────────────┐      UDP       ┌─────────────┐
│  Client Llama   │ ←─────────────────→ │   MCP Server    │ ←──────────────→ │   Arduino   │
│  (Llama 3.2)    │     Port 8000       │   (FastMCP)     │   Port 8888      │  (8 relays) │
└─────────────────┘                     └─────────────────┘                  └─────────────┘
```

## 📋 Prérequis

- Python 3.10+
- GPU NVIDIA avec CUDA (optionnel mais recommandé)
- Arduino avec 8 relais configuré pour UDP
- 4-8 GB RAM minimum

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <votre-repo>
cd mcp-arduino-relay
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration

Éditer `config.py` pour configurer :

```python
# Arduino
ArduinoConfig.ip = "192.168.12.222"  # IP de votre Arduino
ArduinoConfig.port = 8888            # Port UDP de l'Arduino

# Logging
LoggingConfig.log_level = "INFO"     # DEBUG, INFO, WARNING, ERROR
LoggingConfig.log_file = "./logs/arduino_mcp.log"
```

## 📝 Utilisation

### Afficher la configuration

```bash
python main.py config
```

### Tester la connexion Arduino

```bash
python main.py test-arduino --arduino-ip 192.168.12.222 --arduino-port 8888
```

Commandes de test disponibles :
- `STATUS,0,0` - État de tous les relais
- `SET,1,0` - Allumer le relais 1
- `RESET,1,0` - Éteindre le relais 1
- `SET,2,5000` - Allumer le relais 2 pour 5 secondes

### Démarrer le serveur MCP

Dans un terminal :

```bash
python main.py server
```

Le serveur démarre sur `http://0.0.0.0:8000`

### Démarrer le client interactif

Dans un autre terminal :

```bash
python main.py client
```

Options disponibles :
```bash
# Modèle personnalisé
python main.py client --model meta-llama/Llama-3.2-1B-Instruct

# Device spécifique
python main.py client --device cuda

# Niveau de log
python main.py client --log-level DEBUG
```

## 💬 Exemples d'utilisation

Une fois le client démarré, vous pouvez interagir en langage naturel :

```
👤 You: Turn on relay 3
🤖 Assistant: Success: Relay 3 turned ON

👤 You: Turn on relay 5 for 10 seconds
🤖 Assistant: Success: Relay 5 turned ON (will auto-reset after 10000ms)

👤 You: Turn off relay 3
🤖 Assistant: Success: Relay 3 turned OFF

👤 You: What's the status of relay 1?
🤖 Assistant: Relay status: Relay 1 is OFF
```

## 🛠️ Structure du projet

```
mcp-arduino-relay/
├── main.py                 # Point d'entrée principal
├── server.py               # Serveur MCP (FastMCP)
├── client.py               # Client MCP avec Llama
├── arduino_relays.py       # Communication UDP avec Arduino
├── config.py               # Configuration centralisée
├── requirements.txt        # Dépendances Python
├── README.md              # Cette documentation
└── logs/                  # Fichiers de logs (créé automatiquement)
```

## 🔧 Outils MCP disponibles

### `command_relays`

Contrôle un relais spécifique.

**Paramètres :**
- `command` : "SET" (allumer) ou "RESET" (éteindre)
- `relay_index` : Numéro du relais (1-8)
- `auto_reset_delay` : Délai auto-extinction en ms (0-10000)

**Exemple :**
```json
{
  "tool": "command_relays",
  "arguments": {
    "command": "SET",
    "relay_index": 3,
    "auto_reset_delay": 5000
  }
}
```

### `get_relay_status`

Obtient l'état d'un ou plusieurs relais.

**Paramètres :**
- `relay_index` : Numéro du relais (1-8) ou 0 pour tous

**Exemple :**
```json
{
  "tool": "get_relay_status",
  "arguments": {
    "relay_index": 0
  }
}
```

## 🐛 Dépannage

### Le modèle ne se charge pas

```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Utiliser CPU si pas de GPU
python main.py client --device cpu
```

### Erreur de connexion Arduino

1. Vérifier que l'Arduino est accessible :
```bash
ping 192.168.12.222
```

2. Tester la connexion :
```bash
python main.py test-arduino --arduino-ip 192.168.12.222
```

3. Vérifier les logs :
```bash
tail -f logs/arduino_mcp.log
```

### Le client ne trouve pas les outils

1. Vérifier que le serveur est démarré
2. Vérifier l'URL de connexion dans les logs
3. Tester manuellement : `curl http://localhost:8000/mcp`

## 📊 Logs

Les logs sont stockés dans `./logs/arduino_mcp.log` avec rotation automatique (10 MB max, 5 fichiers).

Niveaux de log :
- `DEBUG` : Détails de chaque opération
- `INFO` : Opérations importantes
- `WARNING` : Avertissements
- `ERROR` : Erreurs

## 🔒 Sécurité

⚠️ **Important** : Cette application contrôle des relais physiques. Assurez-vous :

1. De tester d'abord avec des charges non critiques
2. D'implémenter des mécanismes de sécurité sur l'Arduino
3. De limiter l'accès réseau au serveur MCP
4. De valider toutes les commandes avant exécution

## 📚 Ressources

- [MCP Protocol](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Llama 3.2 Models](https://huggingface.co/meta-llama)
- [Arduino UDP Library](https://www.arduino.cc/en/Reference/WiFiUDP)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -am 'Ajout fonctionnalité'`)
4. Push vers