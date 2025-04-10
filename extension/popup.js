document.addEventListener('DOMContentLoaded', function() {
  const tweetsContainer = document.getElementById('tweets-container');
  
  // Function to create a tweet element
  function createTweetElement(tweet) {
    const tweetDiv = document.createElement('div');
    tweetDiv.className = 'tweet';
    
    const header = document.createElement('div');
    header.className = 'tweet-header';
    header.innerHTML = `
      <span class="author-name">${tweet.author.name}</span>
      <span class="author-handle">@${tweet.author.handle}</span>
    `;
    
    const text = document.createElement('div');
    text.className = 'tweet-text';
    text.textContent = tweet.text;
    
    const metrics = document.createElement('div');
    metrics.className = 'tweet-metrics';
    metrics.innerHTML = `
      <span>${tweet.metrics.replies || 0} Replies</span>
      <span>${tweet.metrics.retweets || 0} Retweets</span>
      <span>${tweet.metrics.likes || 0} Likes</span>
    `;
    
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.textContent = new Date(tweet.timestamp).toLocaleString();
    
    tweetDiv.appendChild(header);
    tweetDiv.appendChild(text);
    tweetDiv.appendChild(metrics);
    tweetDiv.appendChild(timestamp);
    
    return tweetDiv;
  }
  
  // Function to update the tweets display
  function updateTweetsDisplay(tweets) {
    if (tweets.length === 0) {
      tweetsContainer.innerHTML = '<div class="no-tweets">No tweets detected yet. Browse X.com to see tweets here.</div>';
      return;
    }
    
    tweetsContainer.innerHTML = '';
    tweets.forEach(tweet => {
      tweetsContainer.appendChild(createTweetElement(tweet));
    });
  }
  
  // Request initial tweets from background script
  chrome.runtime.sendMessage({ action: 'getTweets' }, response => {
    if (response && response.tweets) {
      updateTweetsDisplay(response.tweets);
    }
  });
  
  // Listen for new tweets
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'tweetUpdate') {
      // Add new tweet to the top of the container
      const tweetElement = createTweetElement(message.tweet);
      if (tweetsContainer.firstChild?.className === 'no-tweets') {
        tweetsContainer.innerHTML = '';
      }
      tweetsContainer.insertBefore(tweetElement, tweetsContainer.firstChild);
    }
  });
}); 