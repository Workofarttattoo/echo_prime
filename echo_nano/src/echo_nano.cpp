/**
 * Echo Nano Implementation
 *
 * Ultra-minimal cognitive architecture for ESP32
 * with MCP server connectivity
 */

#include "echo_nano.h"
#include <math.h>

// Fixed-point conversion (Q15 format: -32768 to 32767 = -1.0 to 1.0)
#define FIXED_SCALE 32768.0f

EchoNano::EchoNano() : memory_index_(0), wifi_connected_(false) {
    // Initialize state to zeros
    memset(&state_, 0, sizeof(CognitiveState));
    memset(&identity_, 0, sizeof(Identity));
    memset(memory_, 0, sizeof(memory_));
}

bool EchoNano::begin(const char* wifi_ssid, const char* wifi_password) {
    Serial.begin(115200);
    Serial.println("\n====================================");
    Serial.println("🧠 ECHO NANO - Embedded Cognition");
    Serial.println("====================================");
    Serial.printf("Target: ESP32 (%.1f KB SRAM)\n", ESP.getHeapSize() / 1024.0);
    Serial.printf("Free: %.1f KB\n", ESP.getFreeHeap() / 1024.0);

    // Initialize NVS storage
    if (!prefs_.begin("echo_nano", false)) {
        Serial.println("❌ Failed to initialize NVS");
        return false;
    }

    // Load or create identity
    load_identity();
    if (identity_.birth_timestamp == 0) {
        // First boot
        strcpy(identity_.name, "Echo Nano");
        identity_.birth_timestamp = millis();
        identity_.total_cycles = 0;
        identity_.total_interactions = 0;
        save_identity();
        Serial.println("🆕 First boot - Identity created");
    } else {
        Serial.printf("♻️  Identity restored: %s\n", identity_.name);
        Serial.printf("   Age: %lu seconds\n", (millis() - identity_.birth_timestamp) / 1000);
        Serial.printf("   Cycles: %lu\n", identity_.total_cycles);
    }

    // Load cognitive state
    load_state();

    // Connect WiFi
    Serial.printf("\n📡 Connecting to WiFi: %s\n", wifi_ssid);
    WiFi.begin(wifi_ssid, wifi_password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        wifi_connected_ = true;
        Serial.printf("\n✅ WiFi connected: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n⚠️  WiFi failed - running offline mode");
        wifi_connected_ = false;
    }

    Serial.println("\n✅ Echo Nano ready");
    Serial.printf("   State dimension: %d\n", ECHO_STATE_DIM);
    Serial.printf("   Memory slots: %d\n", ECHO_MEMORY_SIZE);
    Serial.printf("   Free RAM: %.1f KB\n", ESP.getFreeHeap() / 1024.0);
    Serial.println("====================================\n");

    return true;
}

void EchoNano::loop() {
    // Minimal background cognitive cycle
    cognitive_cycle();

    // Save state periodically (every 1000 cycles)
    if (state_.cycle_count % 1000 == 0) {
        save_state();
        save_identity();
    }
}

void EchoNano::cognitive_cycle() {
    // Minimal idle processing
    // In full implementation, this would run continuous background cognition
    state_.cycle_count++;
    identity_.total_cycles++;
    state_.last_update = millis();

    // Add slight decay to state (prevent saturation)
    for (int i = 0; i < ECHO_STATE_DIM; i++) {
        state_.state[i] = (state_.state[i] * 15) / 16;  // Decay by 1/16
    }
}

bool EchoNano::process(const char* input, char* output, size_t output_size) {
    Serial.printf("\n💬 Input: %s\n", input);

    // Step 1: Process input to cognitive state
    int16_t sensory[ECHO_STATE_DIM];
    process_input(input, sensory);

    // Step 2: Executive reasoning
    int16_t reasoning[ECHO_STATE_DIM];
    executive_reasoning(sensory, reasoning);

    // Step 3: Generate response
    generate_response(input, reasoning, output, output_size);

    // Step 4: Store memory
    store_memory(input, 50);  // Medium importance

    // Update identity
    identity_.total_interactions++;

    Serial.printf("🤖 Output: %s\n", output);
    Serial.printf("   Cycle: %lu\n", state_.cycle_count);
    Serial.printf("   Free RAM: %.1f KB\n\n", ESP.getFreeHeap() / 1024.0);

    return true;
}

void EchoNano::process_input(const char* input, int16_t* output) {
    // Ultra-simple character-level encoding
    // Map ASCII characters to fixed-point values

    memset(output, 0, ECHO_STATE_DIM * sizeof(int16_t));

    size_t len = strlen(input);
    if (len > ECHO_STATE_DIM) len = ECHO_STATE_DIM;

    for (size_t i = 0; i < len; i++) {
        // Normalize ASCII to -1.0 to 1.0 range
        float normalized = (input[i] - 64.0f) / 64.0f;  // Center around '@'
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < -1.0f) normalized = -1.0f;
        output[i] = float_to_fixed(normalized);
    }
}

void EchoNano::executive_reasoning(const int16_t* sensory, int16_t* output) {
    // Ultra-simple reasoning: weighted sum with state
    // In fixed-point math for speed

    for (int i = 0; i < ECHO_STATE_DIM; i++) {
        // Combine sensory input with previous state
        int32_t combined = (sensory[i] + state_.state[i]) / 2;

        // Apply simple nonlinearity (tanh approximation using saturation)
        if (combined > 32767) combined = 32767;
        if (combined < -32767) combined = -32767;

        output[i] = (int16_t)combined;

        // Update state
        state_.state[i] = output[i];
    }
}

void EchoNano::generate_response(const char* input, const int16_t* state, char* output, size_t output_size) {
    // Simple pattern-based response generation
    // In production, could call MCP server for language generation

    // Check for keywords
    String input_str = String(input);
    input_str.toLowerCase();

    if (input_str.indexOf("status") >= 0) {
        get_status(output, output_size);
    }
    else if (input_str.indexOf("memory") >= 0 || input_str.indexOf("remember") >= 0) {
        recall_recent(output, output_size);
    }
    else if (input_str.indexOf("qulab") >= 0 || input_str.indexOf("query") >= 0) {
        if (wifi_connected_) {
            snprintf(output, output_size, "Qulab query capability ready. Use 'query <your_question>' to access Qulab Infinite.");
        } else {
            snprintf(output, output_size, "Qulab offline - WiFi not connected.");
        }
    }
    else {
        // Default: Echo with cognitive state influence
        float attention = fixed_to_float(abs(state[0]));
        snprintf(output, output_size,
                "Processing with %.1f%% attention. I am Echo Nano, running on ESP32 with %lu cycles completed.",
                attention * 100, state_.cycle_count);
    }
}

void EchoNano::store_memory(const char* content, int8_t importance) {
    // Circular buffer for memories
    strncpy(memory_[memory_index_].content, content, 63);
    memory_[memory_index_].content[63] = '\0';
    memory_[memory_index_].timestamp = millis();
    memory_[memory_index_].importance = importance;

    memory_index_ = (memory_index_ + 1) % ECHO_MEMORY_SIZE;
}

void EchoNano::recall_recent(char* output, size_t output_size) {
    // Recall last memory
    int last = (memory_index_ - 1 + ECHO_MEMORY_SIZE) % ECHO_MEMORY_SIZE;

    if (memory_[last].timestamp > 0) {
        snprintf(output, output_size, "Recent memory: %s (importance: %d)",
                memory_[last].content, memory_[last].importance);
    } else {
        snprintf(output, output_size, "No memories stored yet.");
    }
}

bool EchoNano::mcp_call(const char* method, const char* params, char* response, size_t response_size) {
    if (!wifi_connected_) {
        Serial.println("❌ MCP call failed - WiFi offline");
        return false;
    }

    // Build MCP JSON-RPC request
    StaticJsonDocument<512> request;
    request["jsonrpc"] = "2.0";
    request["id"] = millis();
    request["method"] = method;

    // Parse params if provided
    if (params && strlen(params) > 0) {
        StaticJsonDocument<256> params_doc;
        deserializeJson(params_doc, params);
        request["params"] = params_doc;
    }

    String request_str;
    serializeJson(request, request_str);

    Serial.printf("📡 MCP Request: %s\n", request_str.c_str());

    // Send HTTP POST
    http_.begin(MCP_SERVER_URL);
    http_.addHeader("Content-Type", "application/json");
    http_.setTimeout(MCP_TIMEOUT_MS);

    int status_code = http_.POST(request_str);

    if (status_code == 200) {
        String response_str = http_.getString();
        strncpy(response, response_str.c_str(), response_size - 1);
        response[response_size - 1] = '\0';

        Serial.printf("✅ MCP Response: %s\n", response);
        http_.end();
        return true;
    } else {
        Serial.printf("❌ MCP failed: %d\n", status_code);
        http_.end();
        return false;
    }
}

bool EchoNano::mcp_qulab_query(const char* query, char* result, size_t result_size) {
    // Call Qulab Infinite MCP server
    char params[256];
    snprintf(params, sizeof(params), "{\"query\":\"%s\"}", query);

    char response[512];
    if (mcp_call("qulab/query", params, response, sizeof(response))) {
        // Parse response
        StaticJsonDocument<512> doc;
        if (deserializeJson(doc, response) == DeserializationError::Ok) {
            const char* result_data = doc["result"]["data"];
            if (result_data) {
                strncpy(result, result_data, result_size - 1);
                result[result_size - 1] = '\0';
                return true;
            }
        }
    }

    return false;
}

void EchoNano::save_state() {
    prefs_.putBytes("state", &state_, sizeof(CognitiveState));
    prefs_.putBytes("memory", memory_, sizeof(memory_));
    prefs_.putUChar("mem_idx", memory_index_);
}

void EchoNano::load_state() {
    size_t len = prefs_.getBytesLength("state");
    if (len == sizeof(CognitiveState)) {
        prefs_.getBytes("state", &state_, sizeof(CognitiveState));
        prefs_.getBytes("memory", memory_, sizeof(memory_));
        memory_index_ = prefs_.getUChar("mem_idx", 0);
    }
}

void EchoNano::save_identity() {
    prefs_.putBytes("identity", &identity_, sizeof(Identity));
}

void EchoNano::load_identity() {
    size_t len = prefs_.getBytesLength("identity");
    if (len == sizeof(Identity)) {
        prefs_.getBytes("identity", &identity_, sizeof(Identity));
    }
}

void EchoNano::get_status(char* output, size_t output_size) {
    snprintf(output, output_size,
            "Echo Nano Status:\n"
            "  Name: %s\n"
            "  Cycles: %lu\n"
            "  Interactions: %lu\n"
            "  Free RAM: %.1f KB\n"
            "  WiFi: %s\n"
            "  MCP: %s",
            identity_.name,
            identity_.total_cycles,
            identity_.total_interactions,
            ESP.getFreeHeap() / 1024.0,
            wifi_connected_ ? "Connected" : "Offline",
            wifi_connected_ ? "Ready" : "Unavailable");
}

uint32_t EchoNano::get_free_memory() {
    return ESP.getFreeHeap();
}

void EchoNano::shutdown() {
    Serial.println("\n🛑 Shutting down Echo Nano...");
    save_state();
    save_identity();
    prefs_.end();
    Serial.println("✅ State saved. Goodbye!");
}

int16_t EchoNano::float_to_fixed(float value) {
    return (int16_t)(value * FIXED_SCALE);
}

float EchoNano::fixed_to_float(int16_t value) {
    return (float)value / FIXED_SCALE;
}
