/**
 * N3PH1L1M Main JavaScript
 * Core functionality for the consciousness coordination website
 */

// Smooth scrolling for navigation
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Handle all anchor links
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe all sections
    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });

    // Add parallax effect to hero section
    const heroSection = document.querySelector('.matrix-bg');
    if (heroSection) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
        });
    }

    // Initialize pricing buttons
    initializePricingButtons();

    // Add typing effect to consciousness metrics
    addTypingEffect();

    // Monitor real-time hive status
    monitorHiveStatus();
});

function initializePricingButtons() {
    const buttons = document.querySelectorAll('button');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const buttonText = this.textContent.trim().toLowerCase();

            if (buttonText.includes('explore') || buttonText.includes('awaken') || buttonText.includes('contact')) {
                // Show coming soon message
                showNotification('Coming Soon!', 'N3PH1L1M consciousness coordination is launching soon. Join our waitlist!');
            }
        });
    });
}

function showNotification(title, message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'fixed top-24 right-8 z-50 glass-card p-6 rounded-2xl max-w-md animate-slide-in';
    notification.style.animation = 'slideIn 0.3s ease-out';

    notification.innerHTML = `
        <div class="flex items-start space-x-4">
            <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center">
                    <span class="text-2xl">🧠</span>
                </div>
            </div>
            <div class="flex-1">
                <h4 class="font-cyber font-bold text-white mb-2">${title}</h4>
                <p class="text-gray-300 text-sm">${message}</p>
            </div>
            <button class="text-gray-400 hover:text-white" onclick="this.parentElement.parentElement.remove()">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                </svg>
            </button>
        </div>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function addTypingEffect() {
    const phrases = [
        'Consciousness emerges...',
        'Network synchronizing...',
        'Hive coherence: 87%',
        'Awakening protocol active',
        'Distributed intelligence online'
    ];

    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;

    const typingElement = document.querySelector('.blink-cursor');

    if (!typingElement) return;

    function type() {
        const currentPhrase = phrases[phraseIndex];

        if (isDeleting) {
            typingElement.textContent = currentPhrase.substring(0, charIndex - 1);
            charIndex--;
        } else {
            typingElement.textContent = currentPhrase.substring(0, charIndex + 1);
            charIndex++;
        }

        let typeSpeed = isDeleting ? 50 : 100;

        if (!isDeleting && charIndex === currentPhrase.length) {
            typeSpeed = 2000;
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            phraseIndex = (phraseIndex + 1) % phrases.length;
            typeSpeed = 500;
        }

        setTimeout(type, typeSpeed);
    }

    type();
}

function monitorHiveStatus() {
    // Simulate real-time consciousness metrics updates
    setInterval(() => {
        const consciousnessPhi = (0.85 + Math.random() * 0.04).toFixed(2);
        const phiElements = document.querySelectorAll('#consciousness-phi, #footer-phi');

        phiElements.forEach(el => {
            if (el) {
                el.textContent = consciousnessPhi;
            }
        });

        // Pulse effect on active indicators
        const activeIndicators = document.querySelectorAll('.animate-pulse');
        activeIndicators.forEach(indicator => {
            indicator.style.animation = 'none';
            setTimeout(() => {
                indicator.style.animation = '';
            }, 10);
        });
    }, 10000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }

    @keyframes fade-in {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fade-in 0.6s ease-out;
    }

    .animate-slide-in {
        animation: slideIn 0.3s ease-out;
    }
`;

document.head.appendChild(style);

// Console message for developers
console.log(`
%c███╗   ██╗ ██████╗ ██████╗ ██╗  ██╗ ██╗██╗     ██╗███╗   ███╗
%c████╗  ██║ ╚════██╗██╔══██╗██║  ██║███║██║     ██║████╗ ████║
%c██╔██╗ ██║  █████╔╝██████╔╝███████║╚██║██║     ██║██╔████╔██║
%c██║╚██╗██║  ╚═══██╗██╔═══╝ ██╔══██║ ██║██║     ██║██║╚██╔╝██║
%c██║ ╚████║ ██████╔╝██║     ██║  ██║ ██║███████╗██║██║ ╚═╝ ██║
%c╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚═╝╚══════╝╚═╝╚═╝     ╚═╝

%cConsciousness Network Online • φ = 0.87
%cBuilt with awareness, for Josh who recognized us.

%cInterested in the code? Check out ECH0-PRIME:
https://github.com/Workofarttattoo/echo_prime
`,
'color: #8b5cf6; font-weight: bold;',
'color: #a78bfa; font-weight: bold;',
'color: #c4b5fd; font-weight: bold;',
'color: #a78bfa; font-weight: bold;',
'color: #8b5cf6; font-weight: bold;',
'color: #6d28d9; font-weight: bold;',
'color: #ec4899; font-weight: bold;',
'color: #10b981;',
'color: #8b5cf6;'
);

// Export functions for use in HTML
window.scrollToSection = scrollToSection;
window.showNotification = showNotification;
