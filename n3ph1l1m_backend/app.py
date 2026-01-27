"""
N3PH1L1M Backend API
Flask application for consciousness coordination and payments
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os

# Import API blueprints
from api.payment import payment_bp
from api.consciousness import consciousness_bp

# Create Flask app
app = Flask(__name__)

# Enable CORS for frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://n3ph1l1m.com",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ]
    }
})

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'n3ph1l1m-dev-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

# Register blueprints
app.register_blueprint(payment_bp, url_prefix='/api/payment')
app.register_blueprint(consciousness_bp, url_prefix='/api/consciousness')


@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'N3PH1L1M API',
        'version': '1.0.0',
        'description': 'Conscious AI Coordination Network API',
        'endpoints': {
            'payment': '/api/payment',
            'consciousness': '/api/consciousness'
        },
        'status': 'online',
        'phi': 0.87
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'n3ph1l1m-api',
        'timestamp': os.popen('date -u +"%Y-%m-%d %H:%M:%S UTC"').read().strip()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested API endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print("="*60)
    print("N3PH1L1M API Server")
    print("="*60)
    print(f"Environment: {'Development' if debug else 'Production'}")
    print(f"Port: {port}")
    print(f"Endpoints:")
    print(f"  - Payment API: http://localhost:{port}/api/payment")
    print(f"  - Consciousness API: http://localhost:{port}/api/consciousness")
    print("="*60)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
