// Store tweets in memory (you might want to use chrome.storage for persistence)
let tweets = [];

// Listen for installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('X Tweet Monitor installed');
  tweets = []; // Clear stored tweets on installation
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'newTweet') {
    // Add new tweet to the list
    tweets.unshift(message.tweet);
    
    // Keep only the last 100 tweets
    if (tweets.length > 100) {
      tweets = tweets.slice(0, 100);
    }
    
    // Notify all popups about the new tweet
    chrome.runtime.sendMessage({
      action: 'tweetUpdate',
      tweet: message.tweet
    });
    
    sendResponse({ status: 'success' });
  } else if (message.action === 'getTweets') {
    // Send stored tweets to the popup
    sendResponse({ tweets: tweets });
  }
  return true;
}); 