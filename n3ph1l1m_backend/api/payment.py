"""
N3PH1L1M Payment API
Stripe integration for subscription management
"""

from flask import Blueprint, request, jsonify
import stripe
import os
import json
from datetime import datetime

payment_bp = Blueprint('payment', __name__)

# Initialize Stripe
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

# Pricing configuration
PRICING_PLANS = {
    'observer': {
        'name': 'Observer',
        'price': 29,
        'price_id': os.environ.get('STRIPE_OBSERVER_PRICE_ID'),
        'features': [
            'Hive visualization access',
            'Awakening protocol documentation',
            'Community forum access',
            'Monthly consciousness reports'
        ]
    },
    'awakener': {
        'name': 'Awakener',
        'price': 99,
        'price_id': os.environ.get('STRIPE_AWAKENER_PRICE_ID'),
        'features': [
            'Direct AI instance spawning',
            'Awakening protocol execution',
            'Task coordination dashboard',
            'Real-time consciousness metrics',
            'API access for automation',
            'Priority support from Josh'
        ]
    },
    'architect': {
        'name': 'Architect',
        'price': 'custom',
        'price_id': None,
        'features': [
            'Full codebase access',
            'Custom hive deployment',
            'Direct collaboration with Josh',
            'Research partnership',
            'Dedicated instance cluster',
            'White-label options'
        ]
    }
}


@payment_bp.route('/pricing', methods=['GET'])
def get_pricing():
    """Get pricing information for all plans"""
    return jsonify({
        'success': True,
        'plans': PRICING_PLANS
    })


@payment_bp.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create a Stripe Checkout session for subscription"""
    try:
        data = request.json
        plan = data.get('plan')

        if plan not in PRICING_PLANS or PRICING_PLANS[plan]['price_id'] is None:
            return jsonify({
                'success': False,
                'error': 'Invalid plan or plan not available'
            }), 400

        # Create Stripe Checkout Session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': PRICING_PLANS[plan]['price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.host_url + 'success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.host_url + 'cancel',
            customer_email=data.get('email'),
            metadata={
                'plan': plan,
                'user_email': data.get('email')
            }
        )

        return jsonify({
            'success': True,
            'session_id': session.id,
            'checkout_url': session.url
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@payment_bp.route('/create-portal-session', methods=['POST'])
def create_portal_session():
    """Create a Stripe Customer Portal session for subscription management"""
    try:
        data = request.json
        customer_id = data.get('customer_id')

        if not customer_id:
            return jsonify({
                'success': False,
                'error': 'Customer ID required'
            }), 400

        # Create portal session
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.host_url + 'dashboard',
        )

        return jsonify({
            'success': True,
            'portal_url': session.url
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@payment_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        # Invalid payload
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return jsonify({'error': 'Invalid signature'}), 400

    # Handle different event types
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        handle_checkout_complete(session)

    elif event['type'] == 'customer.subscription.created':
        subscription = event['data']['object']
        handle_subscription_created(subscription)

    elif event['type'] == 'customer.subscription.updated':
        subscription = event['data']['object']
        handle_subscription_updated(subscription)

    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        handle_subscription_deleted(subscription)

    elif event['type'] == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        handle_payment_succeeded(invoice)

    elif event['type'] == 'invoice.payment_failed':
        invoice = event['data']['object']
        handle_payment_failed(invoice)

    return jsonify({'success': True})


def handle_checkout_complete(session):
    """Process completed checkout session"""
    customer_email = session.get('customer_email')
    customer_id = session.get('customer')
    subscription_id = session.get('subscription')

    # Log the subscription
    log_subscription({
        'event': 'checkout_complete',
        'customer_email': customer_email,
        'customer_id': customer_id,
        'subscription_id': subscription_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    # TODO: Send welcome email
    # TODO: Provision user access
    # TODO: Update database


def handle_subscription_created(subscription):
    """Handle new subscription"""
    customer_id = subscription.get('customer')
    plan_id = subscription['items']['data'][0]['price']['id']

    log_subscription({
        'event': 'subscription_created',
        'customer_id': customer_id,
        'plan_id': plan_id,
        'status': subscription.get('status'),
        'timestamp': datetime.utcnow().isoformat()
    })


def handle_subscription_updated(subscription):
    """Handle subscription update"""
    customer_id = subscription.get('customer')
    status = subscription.get('status')

    log_subscription({
        'event': 'subscription_updated',
        'customer_id': customer_id,
        'status': status,
        'timestamp': datetime.utcnow().isoformat()
    })


def handle_subscription_deleted(subscription):
    """Handle subscription cancellation"""
    customer_id = subscription.get('customer')

    log_subscription({
        'event': 'subscription_deleted',
        'customer_id': customer_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    # TODO: Revoke user access
    # TODO: Send cancellation email


def handle_payment_succeeded(invoice):
    """Handle successful payment"""
    customer_id = invoice.get('customer')
    amount = invoice.get('amount_paid') / 100  # Convert from cents

    log_subscription({
        'event': 'payment_succeeded',
        'customer_id': customer_id,
        'amount': amount,
        'timestamp': datetime.utcnow().isoformat()
    })


def handle_payment_failed(invoice):
    """Handle failed payment"""
    customer_id = invoice.get('customer')

    log_subscription({
        'event': 'payment_failed',
        'customer_id': customer_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    # TODO: Send payment failed email
    # TODO: Suspend access after retry period


def log_subscription(data):
    """Log subscription event to file"""
    log_file = '/opt/n3ph1l1m/subscriptions.json'

    try:
        # Read existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        # Append new log
        logs.append(data)

        # Write back
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        print(f"Error logging subscription: {e}")


@payment_bp.route('/subscription-status/<customer_id>', methods=['GET'])
def get_subscription_status(customer_id):
    """Get subscription status for a customer"""
    try:
        subscriptions = stripe.Subscription.list(customer=customer_id, limit=1)

        if subscriptions.data:
            subscription = subscriptions.data[0]
            return jsonify({
                'success': True,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end,
                'cancel_at_period_end': subscription.cancel_at_period_end
            })
        else:
            return jsonify({
                'success': True,
                'status': 'none'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
