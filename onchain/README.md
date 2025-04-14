# Onchain Module

The `onchain` module contains all the classes and utilities for interacting with blockchain data and protocols. It provides functionality for fetching and analyzing on-chain data related to tokens, pools, portfolios, and analytics.

## Structure

- `analytics/`: Contains analytics-related functionality for analyzing on-chain data
- `pools/`: Classes and utilities for interacting with liquidity pools and AMMs
- `portfolio/`: Portfolio management and analysis tools
- `tokens/`: Token-related functionality including price feeds, metadata, and token analysis

## Purpose

This module serves as the data layer for the OpenQuant platform, providing:
- Real-time and historical on-chain data access
- Protocol interaction capabilities
- Data analysis and processing tools
- Portfolio tracking and management
- Token and pool analytics

## Usage

The module is primarily used by the agent system to gather and analyze on-chain data for making informed decisions about DeFi operations. It provides the foundational data layer that powers the analytics and investment agents. 