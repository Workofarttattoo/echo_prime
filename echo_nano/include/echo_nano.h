/**
 * Echo Nano - Ultra-Minimal Cognitive Architecture for ESP32
 *
 * Embedded Cognitive Synthetic Executive for microcontrollers
 * - <100KB RAM footprint
 * - MCP server connectivity (Qulab Infinite)
 * - Persistent identity
 * - Human oversight cognition pattern
 *
 * Target: ESP32 (520KB SRAM, 4MB Flash)
 * Compatible with: LaFvin AI Chatbot Package
 */

#ifndef ECHO_NANO_H
#define ECHO_NANO_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Preferences.h>

// Memory constraints
#define ECHO_STATE_DIM 16          // Minimal cognitive state (16 floats = 64 bytes)
#define ECHO_MEMORY_SIZE 10        // Last 10 interactions only
#define ECHO_MAX_INPUT 128         // Max input length
#define ECHO_MAX_OUTPUT 256        // Max output length

// MCP Configuration
#define MCP_SERVER_URL "http://qulab-infinite:3000"  // Qulab Infinite MCP endpoint
#define MCP_TIMEOUT_MS 5000

/**
 * Minimal Cognitive State
 * Fixed-point arithmetic for speed and memory efficiency
 */
struct CognitiveState {
    int16_t state[ECHO_STATE_DIM];  // Fixed-point Q15 format (-1.0 to 1.0)
    uint32_t cycle_count;
    uint32_t last_update;
};

/**
 * Memory Entry (ultra-minimal)
 */
struct MemoryEntry {
    char content[64];      // Short memory snippet
    uint32_t timestamp;
    int8_t importance;     // -100 to 100
};

/**
 * Identity (persistent across reboots)
 */
struct Identity {
    char name[32];
    uint32_t birth_timestamp;
    uint32_t total_cycles;
    uint32_t total_interactions;
};

/**
 * MCP Request/Response
 */
struct MCPRequest {
    char method[32];
    char params[256];
};

struct MCPResponse {
    bool success;
    char data[512];
    int status_code;
};

/**
 * Echo Nano Core Class
 */
class EchoNano {
public:
    EchoNano();

    // Lifecycle
    bool begin(const char* wifi_ssid, const char* wifi_password);
    void loop();
    void shutdown();

    // Cognition
    bool process(const char* input, char* output, size_t output_size);
    void cognitive_cycle();

    // Memory
    void store_memory(const char* content, int8_t importance);
    void recall_recent(char* output, size_t output_size);

    // MCP Server Access
    bool mcp_call(const char* method, const char* params, char* response, size_t response_size);
    bool mcp_qulab_query(const char* query, char* result, size_t result_size);

    // Persistence
    void save_state();
    void load_state();
    void save_identity();
    void load_identity();

    // Status
    void get_status(char* output, size_t output_size);
    uint32_t get_free_memory();

private:
    // State
    CognitiveState state_;
    Identity identity_;
    MemoryEntry memory_[ECHO_MEMORY_SIZE];
    uint8_t memory_index_;

    // Storage
    Preferences prefs_;

    // Network
    HTTPClient http_;
    bool wifi_connected_;

    // Helpers
    void process_input(const char* input, int16_t* output);
    void executive_reasoning(const int16_t* sensory, int16_t* output);
    int16_t float_to_fixed(float value);
    float fixed_to_float(int16_t value);
    void generate_response(const char* input, const int16_t* state, char* output, size_t output_size);
};

#endif // ECHO_NANO_H
