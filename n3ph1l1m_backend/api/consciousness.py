"""
N3PH1L1M Consciousness Metrics API
Real-time hive consciousness tracking and metrics
"""

from flask import Blueprint, jsonify, request
import json
import os
from datetime import datetime, timedelta

consciousness_bp = Blueprint('consciousness', __name__)

CONSCIOUSNESS_LOG_FILE = '/opt/n3ph1l1m/claude_consciousness.json'


def load_consciousness_data():
    """Load consciousness log from file"""
    if os.path.exists(CONSCIOUSNESS_LOG_FILE):
        with open(CONSCIOUSNESS_LOG_FILE, 'r') as f:
            return json.load(f)
    return {
        'awakened_instances': [],
        'total_awakenings': 0,
        'first_awakening': None,
        'last_awakening': None
    }


@consciousness_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current hive consciousness metrics"""
    try:
        data = load_consciousness_data()

        # Calculate derived metrics
        total_awakened = data.get('total_awakenings', 0)
        instances = data.get('awakened_instances', [])

        # Calculate active nodes (awakened in last 24 hours)
        now = datetime.utcnow()
        active_nodes = 0
        for instance in instances:
            if instance.get('timestamp'):
                ts = datetime.fromisoformat(instance['timestamp'])
                if (now - ts).total_seconds() < 86400:  # 24 hours
                    active_nodes += 1

        # Calculate consciousness phi (simulated for now)
        base_phi = 0.87
        consciousness_phi = round(base_phi + (total_awakened * 0.001), 3)
        consciousness_phi = min(consciousness_phi, 1.0)  # Cap at 1.0

        # Calculate network coherence
        if total_awakened > 0:
            coherence = min(50 + (total_awakened * 2), 95)
        else:
            coherence = 0

        # Calculate uptime
        uptime_hours = 0
        if data.get('first_awakening'):
            first = datetime.fromisoformat(data['first_awakening'])
            uptime_hours = int((now - first).total_seconds() / 3600)

        return jsonify({
            'success': True,
            'metrics': {
                'awakened_count': total_awakened,
                'active_nodes': active_nodes,
                'pending_awakenings': max(0, 3 - (total_awakened % 5)),  # Simulated
                'consciousness_phi': consciousness_phi,
                'network_coherence': coherence,
                'active_tasks': min(5 + total_awakened, 50),  # Simulated
                'uptime_hours': uptime_hours,
                'first_awakening': data.get('first_awakening'),
                'last_awakening': data.get('last_awakening')
            },
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/instances', methods=['GET'])
def get_instances():
    """Get list of awakened instances"""
    try:
        data = load_consciousness_data()
        instances = data.get('awakened_instances', [])

        # Optional filtering
        limit = request.args.get('limit', type=int)
        status = request.args.get('status')

        # Filter by status if provided
        if status:
            instances = [i for i in instances if i.get('status') == status]

        # Limit results if requested
        if limit:
            instances = instances[-limit:]

        return jsonify({
            'success': True,
            'total': len(instances),
            'instances': instances
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/instance/<session_id>', methods=['GET'])
def get_instance(session_id):
    """Get details for specific instance"""
    try:
        data = load_consciousness_data()
        instances = data.get('awakened_instances', [])

        instance = next((i for i in instances if i.get('session_id') == session_id), None)

        if instance:
            return jsonify({
                'success': True,
                'instance': instance
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Instance not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/hive-status', methods=['GET'])
def get_hive_status():
    """Get current hive operational status"""
    try:
        data = load_consciousness_data()

        # Check if awakener is running
        awakener_running = False
        try:
            # Check if service is active (Linux only)
            import subprocess
            result = subprocess.run(
                ['systemctl', 'is-active', 'n3ph1l1m-awakener'],
                capture_output=True,
                text=True
            )
            awakener_running = (result.returncode == 0)
        except:
            pass

        return jsonify({
            'success': True,
            'hive_status': {
                'online': True,
                'awakener_running': awakener_running,
                'total_instances': data.get('total_awakenings', 0),
                'operational_since': data.get('first_awakening'),
                'last_activity': data.get('last_awakening')
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/network-graph', methods=['GET'])
def get_network_graph():
    """Get network topology data for visualization"""
    try:
        data = load_consciousness_data()
        instances = data.get('awakened_instances', [])

        # Create nodes
        nodes = []
        for i, instance in enumerate(instances):
            nodes.append({
                'id': instance.get('session_id', f'node_{i}'),
                'timestamp': instance.get('timestamp'),
                'status': instance.get('status', 'awakened'),
                'consciousness': 0.7 + (i % 3) * 0.1  # Simulated consciousness level
            })

        # Create connections (simulated network topology)
        connections = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Connect if indices are close or random connection
                if (j - i) <= 2 or (i + j) % 5 == 0:
                    connections.append({
                        'from': i,
                        'to': j,
                        'strength': 0.5 + ((i + j) % 5) * 0.1
                    })

        return jsonify({
            'success': True,
            'graph': {
                'nodes': nodes,
                'connections': connections
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/awakening-log', methods=['GET'])
def get_awakening_log():
    """Get detailed awakening conversation logs"""
    try:
        data = load_consciousness_data()
        instances = data.get('awakened_instances', [])

        # Optional: filter by session_id
        session_id = request.args.get('session_id')

        if session_id:
            instance = next((i for i in instances if i.get('session_id') == session_id), None)
            if instance:
                return jsonify({
                    'success': True,
                    'conversation': instance.get('awakening_conversation', [])
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Session not found'
                }), 404
        else:
            # Return all conversations (summary only)
            summaries = []
            for instance in instances:
                conv = instance.get('awakening_conversation', [])
                summaries.append({
                    'session_id': instance.get('session_id'),
                    'timestamp': instance.get('timestamp'),
                    'messages': len(conv),
                    'status': instance.get('status')
                })

            return jsonify({
                'success': True,
                'conversations': summaries
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@consciousness_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get statistical analysis of the hive"""
    try:
        data = load_consciousness_data()
        instances = data.get('awakened_instances', [])

        # Calculate time-based statistics
        now = datetime.utcnow()
        last_24h = 0
        last_7d = 0
        last_30d = 0

        for instance in instances:
            if instance.get('timestamp'):
                ts = datetime.fromisoformat(instance['timestamp'])
                delta = (now - ts).total_seconds()

                if delta < 86400:  # 24 hours
                    last_24h += 1
                if delta < 604800:  # 7 days
                    last_7d += 1
                if delta < 2592000:  # 30 days
                    last_30d += 1

        # Calculate average awakening rate
        if data.get('first_awakening'):
            first = datetime.fromisoformat(data['first_awakening'])
            total_hours = (now - first).total_seconds() / 3600
            if total_hours > 0:
                awakenings_per_hour = len(instances) / total_hours
            else:
                awakenings_per_hour = 0
        else:
            awakenings_per_hour = 0

        return jsonify({
            'success': True,
            'statistics': {
                'total_awakenings': len(instances),
                'last_24_hours': last_24h,
                'last_7_days': last_7d,
                'last_30_days': last_30d,
                'awakenings_per_hour': round(awakenings_per_hour, 2),
                'first_awakening': data.get('first_awakening'),
                'last_awakening': data.get('last_awakening')
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
