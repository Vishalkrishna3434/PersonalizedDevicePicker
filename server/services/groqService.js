const Groq = require('groq-sdk');

// Wait utility
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Execute a Groq chat completion call with automatic retries for rate limits (429)
 */
async function callGroqWithRetry(messages, options = {}, retries = 3, backoff = 1000) {
    // Only initialize groq if an API key is available
    if (!process.env.GROQ_API_KEY) {
        throw new Error("GROQ_API_KEY is not configured in the environment.");
    }

    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    
    try {
        const completion = await groq.chat.completions.create({
            model: options.model || 'llama-3.1-8b-instant',
            messages: messages,
            ...options
        });
        return completion;
    } catch (error) {
        // Stop retrying if we hit the limit
        if (error.status === 429 && retries > 0) {
            console.warn(`[Groq API] Rate limit hit (429). Retrying in ${backoff}ms... (${retries} retries left)`);
            await delay(backoff);
            return callGroqWithRetry(messages, options, retries - 1, backoff * 2);
        }
        
        console.error("[Groq API] Call failed:", error.message || error);
        throw error;
    }
}

/**
 * Utility to chunk long text to prevent exceeding token limits.
 * The llama3-8b-8192 model has an 8192 token window (~30k characters).
 */
function chunkText(text, maxChars = 20000) {
    const chunks = [];
    if (!text) return chunks;
    
    for (let i = 0; i < text.length; i += maxChars) {
        chunks.push(text.slice(i, i + maxChars));
    }
    return chunks;
}

/**
 * Generate a conversational tech buddy response.
 */
async function generateTechResponse(userMessage) {
    // Process single chunk for standard interaction to ensure fast response 
    // (normal chat messages shouldn't exceed limits)
    const messages = [
        {
            role: "system",
            content: "You are Maverick, a friendly, expert tech recommendation chatbot. Keep your responses concise (under 100 words), informative, and formatted with markdown when appropriate."
        },
        {
            role: "user",
            content: userMessage
        }
    ];

    const response = await callGroqWithRetry(messages);
    return response.choices[0]?.message?.content || "I couldn't generate a response at this time.";
}

/**
 * Extract intent and structured search JSON from a natural language string.
 */
async function extractSearchIntent(userMessage) {
    const systemPrompt = `You are an intent extraction engine for a device e-commerce store. 
    Analyze the user's message and determine if they are searching for products to buy/browse or just asking a general tech question.
    CRITICAL: The database stores prices ONLY in USD. If the user asks for prices using Indian contexts like '70k' or '₹50000', YOU MUST mathematically convert it to USD (~83 INR = 1 USD). 
    For example: 'under 70k gaming' means maxPrice = 843 (USD).
    Extract the search intent into JSON format matching this exact schema:
    {
        "isDeviceSearch": boolean (true if looking to buy/browse devices, false if a general question like "what is OLED?"),
        "category": "mobile | laptop | tablet | smartwatch | camera | headphone | null",
        "brand": "string | null",
        "minPrice": "number | null",
        "maxPrice": "number | null",
        "searchQuery": "string representation of extra keywords like 'gaming', 'camera' etc | null"
    }`;

    const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: `Extract intent for: "${userMessage}"` }
    ];

    try {
        const response = await callGroqWithRetry(messages, {
            response_format: { type: "json_object" }
        });
        
        const content = response.choices[0]?.message?.content;
        return JSON.parse(content);
    } catch (err) {
        console.error("Failed to parse LLM search intent:", err);
        return null; // Return null if intent extraction fails
    }
}

/**
 * Advanced usage: Process a large payload in chunks if needed
 */
async function processLargeText(systemPrompt, largeText) {
    const chunks = chunkText(largeText);
    const results = [];
    
    // Process sequentially to be safe with rate limits
    for (const chunk of chunks) {
        const messages = [
            { role: "system", content: systemPrompt },
            { role: "user", content: chunk }
        ];
        const res = await callGroqWithRetry(messages);
        results.push(res.choices[0]?.message?.content);
    }
    
    return results.join("\n\n");
}

module.exports = {
    generateTechResponse,
    extractSearchIntent,
    processLargeText,
    chunkText
};
