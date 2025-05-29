<p align="center">
  <img src="https://raw.githubusercontent.com/OpenGradient/public_images/refs/heads/main/bitquant_readme_banner.png" alt="BitQuant Banner"/>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/OpenGradient/BitQuant" alt="License"></a>
  <a href="https://github.com/OpenGradient/BitQuant/stargazers"><img src="https://img.shields.io/github/stars/OpenGradient/BitQuant?style=social" alt="GitHub stars"></a>
  <a href="https://www.bitquant.io/"><img src="https://img.shields.io/badge/Try%20BitQuant-bitquant.io-blue?logo=chrome-browser" alt="Try BitQuant"></a>
  <a href="https://docs.opengradient.ai/"><img src="https://img.shields.io/badge/Documentation-OpenGradient%20Docs-orange?logo=readthedocs" alt="Documentation"></a>
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

<p align="center">
  <img src="https://raw.githubusercontent.com/OpenGradient/public_images/refs/heads/main/bitquant_architecture.png" alt="BitQuant Architecture Diagram" width="100%"/>
</p>

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

<p align="center">
  <img src="https://raw.githubusercontent.com/OpenGradient/public_images/refs/heads/main/bitquant_example_image_2.png" alt="BitQuant Example Query" width="100%"/>
</p>

You can also try BitQuant instantly on the <a href="https://www.bitquant.io/">production server</a>.

---

## 💡 Sample Questions

Here are some example queries you can try with BitQuant:

### 🏦 DeFi Interactions
- Which protocols are delivering the best risk-adjusted yields right now?
- What's my potential impermanent loss risk if I provide liquidity to the USDC-SOL pool under different market scenarios?
- Calculate a comprehensive risk score for the top 5 Solana DeFi protocols based on TVL trends, code audits, and historical performance
- Compare the TVL growth, volatility, and stability metrics for Kamino vs Orca vs Raydium
- Which lending protocols have maintained the most stable yields over the past 3 months?

### 📊 Portfolio Analytics
- Can you analyze my portfolio's rolling volatility and identify which assets are contributing most to risk?
- How do the volatility trends of my top portfolio assets compare over the last 90 days?
- Show me the correlation between my holdings and provide insights on how to better diversify?
- What's my current portfolio risk assessment and how can I optimize for a better risk-return ratio?
- What's the maximum drawdown for my current portfolio and how does it compare to market benchmarks?

### 📈 Market Insights
- Based on current volatility trends and price patterns, what phase of the market cycle are we likely in?
- Based on historical data, what's the volatility forecast for BTC and ETH in the coming month?

---

## 🧑‍💻 Types of Agents You Can Build

BitQuant is designed to support a wide range of quantitative and DeFi-focused AI agents. Out of the box, the framework includes:

### 1. Analytics Agent
- **Purpose:** Provides deep analytics on portfolios, tokens, protocols, and market trends.
- **Capabilities:**
  - Analyze portfolio volatility, drawdowns, and diversification
  - Evaluate token and protocol risks
  - Track TVL, yield, and performance metrics
  - Identify trends and generate actionable market insights
- **Example Use Cases:**
  - "Analyze my portfolio’s risk profile."
  - "Show TVL trends for Solana DeFi protocols."

### 2. Investor Agent
- **Purpose:** Helps users find and act on yield opportunities and optimize DeFi strategies.
- **Capabilities:**
  - Recommend optimal pools and lending opportunities
  - Compare APRs, TVL, and risk across protocols
  - Guide users through liquidity provision, lending, and yield farming
- **Example Use Cases:**
  - "Which pools offer the best stablecoin yields?"
  - "Compare Kamino and Orca for USDC/SOL."

### 3. Custom Agents
- **Purpose:** The framework is extensible—developers can build agents for:
  - Automated trading strategies
  - On-chain data monitoring and alerting
  - NFT analytics
  - Cross-chain portfolio management
  - Any custom DeFi or analytics workflow

> **Tip:** Agents are modular and can be combined, routed, or extended to suit your specific use case. See the `agent/` directory and templates for examples and customization.

---

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
- Try Bitquant at [bitquant.io](https://www.bitquant.io/)
- OpenGradient Documentation: [OpenGradient Docs](https://docs.opengradient.ai/)
- Join the discussion: [GitHub Discussions](https://github.com/OpenGradient/BitQuant/discussions)
- Support: [BitQuant Discord](https://discord.com/channels/1132794141403791483/1377340212576911410)

---

<p align="center">
  <em>Empowering next-gen quantitative AI agents with OpenGradient.</em>
</p>
