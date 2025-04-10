# Two Ligma Chrome Extension

A basic Chrome Extension skeleton for Two Ligma.

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked" and select the `extension` directory

## Development

The extension consists of the following main components:

- `manifest.json`: Configuration file for the extension
- `popup.html` & `popup.js`: The popup UI and its functionality
- `background.js`: Service worker for background tasks
- `content.js`: Script that runs on web pages
- `icons/`: Directory containing extension icons

## Features

- Basic popup interface
- Content script injection
- Background service worker
- Message passing between components

## Customization

1. Replace the placeholder icons in the `icons/` directory with your own
2. Modify the manifest.json to add additional permissions if needed
3. Update the content.js script to implement your desired webpage modifications
4. Customize the popup.html and popup.js for your specific UI needs

## Testing

1. After making changes, reload the extension in `chrome://extensions/`
2. Click the extension icon to test the popup
3. Open the Chrome DevTools console to see logs from background and content scripts 