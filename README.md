<p align="center">
  <img src="https://placehold.co/900x200?text=BitQuant+by+OpenGradient" alt="BitQuant Banner"/>
</p>

<p align="center">
  <a href="https://github.com/OpenGradient/BitQuant/actions"><img src="https://img.shields.io/github/actions/workflow/status/OpenGradient/BitQuant/ci.yml?branch=main" alt="Build Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/OpenGradient/BitQuant" alt="License"></a>
  <a href="https://pypi.org/project/bitquant/"><img src="https://img.shields.io/pypi/v/bitquant.svg" alt="PyPI version"></a>
  <a href="https://github.com/OpenGradient/BitQuant/stargazers"><img src="https://img.shields.io/github/stars/OpenGradient/BitQuant?style=social" alt="GitHub stars"></a>
</p>

---

# 🚀 BitQuant by OpenGradient

**BitQuant** is a open-source AI agent framework for building quantitative AI agents. It leverages specialized models for ML-powered analytics, trading, portfolio management, and more—all through a natural language interface. BitQuant exposes a REST API that turns user inputs like "What is the current risk profile on Bitcoin?" or "Optimize my portfolio for maximum risk-adjusted returns" into actionable insights.

---

## 📑 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Integrations](#integrations)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## ✨ Features

- 🤖 Build and deploy quantitative AI agents for analytics, trading, and portfolio management
- 🧠 Natural language interface for complex financial queries
- 🔌 Modular architecture with agent and tool plug-ins
- 📈 Real-time crypto analytics and risk profiling
- 🌐 REST API for seamless integration
- ⚡ Fast setup and extensible codebase

## 🏗️ Architecture

```
agent/      # Agent logic and tool definitions
api/        # Server API input/output types
onchain/    # Classes for on-chain data (tokens, pools, etc.)
server/     # Flask server exposing the API
static/     # Static assets for web interface
subnet/     # Bittensor Subnet-related functionality
templates/  # LLM prompt templates for agent
testclient/ # Client for testing the API
testutils/  # Utility functions for testing
```

### Agents

- **Analytics Agent**: Handles crypto analytics (price trends, risks, trending tokens, etc.)
- **Investment Agent**: Helps users select lending/AMM pools to maximize returns on Solana

> The router in `server.py` decides which agent to use for each user query.

## ⚙️ Installation

```bash
make venv
source venv/bin/activate
make install
```

## 🚦 Usage

1. Copy the example environment file and fill out your credentials:
   ```bash
   cp .env.example .env
   # Edit .env and fill in the required values
   ```
2. Run the server:
   ```bash
   make run
   ```
3. (Optional) Try a sample query:
   ```bash
   make sample
   ```

## 🛠️ Configuration

- All configuration is handled via the `.env` file, which you can generate from `.env.example`.
- Fill in all required secrets and keys as described in `.env.example`.

## 🔌 Integrations

- **REST API**: Exposes endpoints for agent interaction
- **Bittensor Subnet**: For decentralized compute
- **Custom LLM Prompts**: In `templates/`

## 🧪 Testing

To run all tests:
```bash
make test
```

## 🚀 Deployment

Build and run in production:
```bash
make docker
make prod
```

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for features, bugs, or documentation improvements.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💬 Contact

- Powered by [OpenGradient](https://opengradient.ai)
-- Join the discussion: [GitHub Discussions](https://github.com/OpenGradient/BitQuant/discussions)

---

<p align="center">
  <em>Empowering next-gen quantitative AI agents with OpenGradient.</em>
</p>
