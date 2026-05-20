/**
 * Echo Nano - LaFvin AI Chatbot Integration
 *
 * Demonstrates Echo cognitive architecture on ESP32
 * with MCP server connectivity to Qulab Infinite
 *
 * Hardware: ESP32 (LaFvin AI Chatbot Package)
 * Features:
 * - Persistent identity across reboots
 * - Minimal cognitive processing (<100KB RAM)
 * - MCP server access for Qulab queries
 * - Human oversight pattern
 */

#include "echo_nano.h"

// WiFi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Echo Nano instance
EchoNano echo;

// Input buffer
char input_buffer[ECHO_MAX_INPUT];
int input_pos = 0;

void setup() {
    // Initialize Echo Nano
    if (!echo.begin(WIFI_SSID, WIFI_PASSWORD)) {
        Serial.println("❌ Echo Nano initialization failed!");
        while(1) delay(1000);
    }

    Serial.println("\n🎮 Echo Nano Interactive Mode");
    Serial.println("Commands:");
    Serial.println("  status    - Show system status");
    Serial.println("  memory    - Recall recent memories");
    Serial.println("  query <q> - Query Qulab Infinite via MCP");
    Serial.println("  Any text  - Cognitive processing\n");
    Serial.print("→ ");
}

void loop() {
    // Background cognitive cycle
    echo.loop();

    // Check for serial input
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (input_pos > 0) {
                input_buffer[input_pos] = '\0';

                // Process input
                processInput(input_buffer);

                // Reset buffer
                input_pos = 0;
                Serial.print("→ ");
            }
        } else if (input_pos < ECHO_MAX_INPUT - 1) {
            input_buffer[input_pos++] = c;
            Serial.print(c);  // Echo character
        }
    }

    delay(10);  // Small delay to avoid busy-wait
}

void processInput(const char* input) {
    Serial.println();  // New line after input

    char output[ECHO_MAX_OUTPUT];

    // Check for Qulab query command
    String input_str = String(input);
    if (input_str.startsWith("query ")) {
        String query = input_str.substring(6);

        Serial.println("🔍 Querying Qulab Infinite via MCP...");

        char result[512];
        if (echo.mcp_qulab_query(query.c_str(), result, sizeof(result))) {
            Serial.printf("📊 Qulab Result:\n%s\n\n", result);

            // Store in memory
            char memory_entry[64];
            snprintf(memory_entry, sizeof(memory_entry), "Queried Qulab: %s", query.c_str());
            echo.store_memory(memory_entry, 80);  // High importance
        } else {
            Serial.println("❌ Qulab query failed\n");
        }
    } else {
        // Regular cognitive processing
        echo.process(input, output, sizeof(output));
    }
}

/**
 * LaFvin Package Integration Notes:
 *
 * 1. MCP Server Configuration:
 *    - Add Qulab Infinite MCP endpoint in LaFvin config
 *    - Set MCP_SERVER_URL to your Qulab instance
 *
 * 2. Memory Usage:
 *    - Echo Nano uses ~80-90KB RAM
 *    - Leaves ~430KB free for LaFvin operations
 *    - Compatible with LaFvin chatbot package
 *
 * 3. Integration Points:
 *    - LaFvin can call echo.process() for cognitive layer
 *    - Echo can use LaFvin's WiFi/BLE connections
 *    - Shared MCP server access
 *
 * 4. Human Oversight:
 *    - All Qulab queries logged to memory
 *    - Persistent audit trail in NVS
 *    - Can be reviewed via 'memory' command
 */
