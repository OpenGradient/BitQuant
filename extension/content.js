// Store the last processed tweet to avoid duplicates
let lastProcessedTweet = null;

// Function to extract tweet information
function extractTweetInfo(tweetElement) {
  try {
    // Find the tweet text
    const tweetText = tweetElement.querySelector('[data-testid="tweetText"]')?.textContent;
    if (!tweetText) return null;

    // Find the author
    const authorElement = tweetElement.querySelector('[data-testid="User-Name"]');
    const authorName = authorElement?.textContent?.split('\n')[0];
    const authorHandle = authorElement?.querySelector('a')?.href?.split('/').pop();

    // Find engagement metrics
    const replyCount = tweetElement.querySelector('[data-testid="reply"]')?.textContent;
    const retweetCount = tweetElement.querySelector('[data-testid="retweet"]')?.textContent;
    const likeCount = tweetElement.querySelector('[data-testid="like"]')?.textContent;

    return {
      text: tweetText,
      author: {
        name: authorName,
        handle: authorHandle
      },
      metrics: {
        replies: replyCount,
        retweets: retweetCount,
        likes: likeCount
      },
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error extracting tweet info:', error);
    return null;
  }
}

// Function to process new tweets
function processNewTweets() {
  const tweets = document.querySelectorAll('[data-testid="tweet"]');
  
  tweets.forEach(tweet => {
    const tweetInfo = extractTweetInfo(tweet);
    if (tweetInfo && tweetInfo !== lastProcessedTweet) {
      lastProcessedTweet = tweetInfo;
      
      // Send tweet info to background script
      chrome.runtime.sendMessage({
        action: 'newTweet',
        tweet: tweetInfo
      });
    }
  });
}

// Set up MutationObserver to detect new tweets
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    if (mutation.addedNodes.length) {
      processNewTweets();
    }
  }
});

// Start observing the document
observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Initial processing
processNewTweets(); 