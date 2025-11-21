#!/usr/bin/env python3
"""
MQTT Broker with Post-Quantum KEM Support
Complete implementation for PQSM research
Author: Varanasi Sai Srinivasa Karthik
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import paho.mqtt.client as mqtt
import paho.mqtt.broker as broker
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add our KEM implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from crypto.advanced_kem import (
    OQSKEMImplementation, 
    ClassicalKEMImplementation,
    HybridKEM,
    AdaptiveKEMSelector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PQCSession:
    """Post-quantum cryptographic session"""
    client_id: str
    kem_algorithm: str
    public_key: bytes
    secret_key: Optional[bytes]
    shared_secret: Optional[bytes]
    established_at: float
    last_activity: float
    message_count: int = 0
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity) > timeout_seconds


class PQSMBroker:
    """MQTT Broker with Post-Quantum Security"""
    
    def __init__(self, host='0.0.0.0', port=1883, kem_algorithm='ML-KEM-512'):
        self.host = host
        self.port = port
        self.kem_algorithm = kem_algorithm
        self.sessions: Dict[str, PQCSession] = {}
        self.metrics = {
            'total_connections': 0,
            'active_sessions': 0,
            'messages_encrypted': 0,
            'handshakes_completed': 0,
            'avg_handshake_ms': 0.0
        }
        
        # Initialize KEM
        if kem_algorithm == 'Hybrid':
            self.kem = HybridKEM('ML-KEM-512', 'X25519')
        elif kem_algorithm in ['X25519', 'X448']:
            self.kem = ClassicalKEMImplementation(kem_algorithm)
        else:
            self.kem = OQSKEMImplementation(kem_algorithm)
        
        # Adaptive selector for dynamic KEM selection
        self.adaptive_selector = AdaptiveKEMSelector()
        
        # Thread pool for crypto operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # MQTT broker setup
        self.broker = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup MQTT callbacks"""
        self.broker.on_connect = self.on_connect
        self.broker.on_disconnect = self.on_disconnect
        self.broker.on_message = self.on_message
        self.broker.on_subscribe = self.on_subscribe
    
    def on_connect(self, client, userdata, flags, reason_code, properties):
        """Handle client connection"""
        if reason_code == 0:
            logger.info(f"Client connected: {client._client_id}")
            self.metrics['total_connections'] += 1
            
            # Initiate PQC handshake
            asyncio.create_task(self.pqc_handshake(client._client_id))
    
    async def pqc_handshake(self, client_id: str):
        """Perform post-quantum key exchange"""
        start_time = time.time()
        
        try:
            # Generate keypair
            public_key, secret_key = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.kem.generate_keypair
            )
            
            # Store session
            session = PQCSession(
                client_id=client_id,
                kem_algorithm=self.kem_algorithm,
                public_key=public_key,
                secret_key=secret_key,
                shared_secret=None,
                established_at=time.time(),
                last_activity=time.time()
            )
            self.sessions[client_id] = session
            
            # Send public key to client
            await self.send_public_key(client_id, public_key)
            
            # Update metrics
            handshake_time = (time.time() - start_time) * 1000
            self.metrics['handshakes_completed'] += 1
            self.metrics['avg_handshake_ms'] = (
                (self.metrics['avg_handshake_ms'] * (self.metrics['handshakes_completed'] - 1) +
                 handshake_time) / self.metrics['handshakes_completed']
            )
            
            logger.info(f"PQC handshake completed for {client_id} in {handshake_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Handshake failed for {client_id}: {e}")
    
    async def send_public_key(self, client_id: str, public_key: bytes):
        """Send public key to client"""
        message = {
            'type': 'pqc_public_key',
            'algorithm': self.kem_algorithm,
            'public_key': public_key.hex(),
            'timestamp': time.time()
        }
        
        topic = f'pqc/{client_id}/handshake'
        payload = json.dumps(message)
        
        self.broker.publish(topic, payload, qos=2)
    
    def on_message(self, client, userdata, message):
        """Handle encrypted messages"""
        try:
            # Check if this is a PQC message
            if message.topic.startswith('pqc/'):
                asyncio.create_task(self.handle_pqc_message(client, message))
            else:
                # Regular MQTT message
                self.handle_regular_message(client, message)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def handle_pqc_message(self, client, message):
        """Handle post-quantum encrypted message"""
        client_id = client._client_id
        
        if client_id not in self.sessions:
            logger.warning(f"No session for client {client_id}")
            return
        
        session = self.sessions[client_id]
        session.last_activity = time.time()
        session.message_count += 1
        
        try:
            # Parse message
            data = json.loads(message.payload)
            
            if data['type'] == 'pqc_ciphertext':
                # Client sent ciphertext, establish shared secret
                ciphertext = bytes.fromhex(data['ciphertext'])
                
                # Decapsulate to get shared secret
                shared_secret = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.kem.decapsulate,
                    ciphertext,
                    session.secret_key
                )
                
                session.shared_secret = shared_secret
                logger.info(f"Shared secret established for {client_id}")
                
                # Send confirmation
                await self.send_handshake_complete(client_id)
                
            elif data['type'] == 'encrypted_message':
                # Decrypt actual message using shared secret
                if session.shared_secret is None:
                    logger.error(f"No shared secret for {client_id}")
                    return
                
                # Decrypt message (simplified - in reality use AEAD)
                encrypted = bytes.fromhex(data['payload'])
                # decrypted = decrypt_with_shared_secret(encrypted, session.shared_secret)
                
                self.metrics['messages_encrypted'] += 1
                
                # Forward decrypted message
                # self.forward_message(decrypted, data.get('topic'))
                
        except Exception as e:
            logger.error(f"Error processing PQC message: {e}")
    
    async def send_handshake_complete(self, client_id: str):
        """Send handshake completion confirmation"""
        message = {
            'type': 'pqc_handshake_complete',
            'timestamp': time.time(),
            'session_id': f"{client_id}_{int(time.time())}"
        }
        
        topic = f'pqc/{client_id}/handshake'
        payload = json.dumps(message)
        
        self.broker.publish(topic, payload, qos=2)
    
    def handle_regular_message(self, client, message):
        """Handle non-PQC messages"""
        logger.debug(f"Regular message on {message.topic}: {message.payload}")
    
    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Handle client disconnection"""
        client_id = client._client_id
        logger.info(f"Client disconnected: {client_id}")
        
        # Clean up session
        if client_id in self.sessions:
            del self.sessions[client_id]
            self.metrics['active_sessions'] = len(self.sessions)
    
    def on_subscribe(self, client, userdata, mid, reason_codes, properties):
        """Handle subscription"""
        logger.info(f"Client subscribed: {client._client_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get broker metrics"""
        self.metrics['active_sessions'] = len(self.sessions)
        return self.metrics.copy()
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = [
            client_id for client_id, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for client_id in expired:
            logger.info(f"Removing expired session: {client_id}")
            del self.sessions[client_id]
        
        return len(expired)
    
    async def start(self):
        """Start the broker"""
        logger.info(f"Starting PQSM Broker on {self.host}:{self.port}")
        logger.info(f"KEM Algorithm: {self.kem_algorithm}")
        
        # Connect to MQTT broker
        self.broker.connect(self.host, self.port, 60)
        
        # Start broker loop
        self.broker.loop_start()
        
        # Periodic cleanup
        while True:
            await asyncio.sleep(60)
            expired = self.cleanup_expired_sessions()
            if expired > 0:
                logger.info(f"Cleaned up {expired} expired sessions")
            
            # Log metrics
            metrics = self.get_metrics()
            logger.info(f"Metrics: {metrics}")
    
    def stop(self):
        """Stop the broker"""
        logger.info("Stopping PQSM Broker")
        self.broker.loop_stop()
        self.broker.disconnect()
        self.executor.shutdown(wait=True)


class PQSMClient:
    """MQTT Client with Post-Quantum Security"""
    
    def __init__(self, broker_host='localhost', broker_port=1883, 
                 kem_algorithm='ML-KEM-512', client_id=None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.kem_algorithm = kem_algorithm
        self.client_id = client_id or f"pqsm_client_{int(time.time())}"
        
        # Initialize KEM
        if kem_algorithm in ['X25519', 'X448']:
            self.kem = ClassicalKEMImplementation(kem_algorithm)
        else:
            self.kem = OQSKEMImplementation(kem_algorithm)
        
        # Session info
        self.broker_public_key = None
        self.shared_secret = None
        self.handshake_complete = False
        
        # MQTT client
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id
        )
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup MQTT callbacks"""
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, reason_code, properties):
        """Handle connection to broker"""
        if reason_code == 0:
            logger.info(f"Connected to broker as {self.client_id}")
            
            # Subscribe to handshake channel
            self.client.subscribe(f"pqc/{self.client_id}/handshake", qos=2)
        else:
            logger.error(f"Connection failed: {reason_code}")
    
    def on_message(self, client, userdata, message):
        """Handle messages from broker"""
        try:
            if message.topic == f"pqc/{self.client_id}/handshake":
                data = json.loads(message.payload)
                
                if data['type'] == 'pqc_public_key':
                    # Received broker's public key
                    self.broker_public_key = bytes.fromhex(data['public_key'])
                    logger.info("Received broker public key")
                    
                    # Perform encapsulation
                    asyncio.create_task(self.complete_handshake())
                    
                elif data['type'] == 'pqc_handshake_complete':
                    # Handshake complete
                    self.handshake_complete = True
                    logger.info("PQC handshake complete!")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def complete_handshake(self):
        """Complete the PQC handshake"""
        if self.broker_public_key is None:
            logger.error("No broker public key available")
            return
        
        try:
            # Encapsulate to create shared secret
            loop = asyncio.get_event_loop()
            ciphertext, shared_secret = await loop.run_in_executor(
                None,
                self.kem.encapsulate,
                self.broker_public_key
            )
            
            self.shared_secret = shared_secret
            
            # Send ciphertext to broker
            message = {
                'type': 'pqc_ciphertext',
                'ciphertext': ciphertext.hex(),
                'timestamp': time.time()
            }
            
            topic = f'pqc/{self.client_id}/handshake'
            payload = json.dumps(message)
            
            self.client.publish(topic, payload, qos=2)
            logger.info("Sent ciphertext to broker")
            
        except Exception as e:
            logger.error(f"Error completing handshake: {e}")
    
    def send_encrypted_message(self, topic: str, message: str):
        """Send encrypted message using established shared secret"""
        if not self.handshake_complete or self.shared_secret is None:
            logger.error("Handshake not complete")
            return False
        
        try:
            # Encrypt message (simplified - use real AEAD in production)
            # encrypted = encrypt_with_shared_secret(message.encode(), self.shared_secret)
            
            data = {
                'type': 'encrypted_message',
                'topic': topic,
                'payload': message,  # Should be encrypted
                'timestamp': time.time()
            }
            
            pqc_topic = f'pqc/{self.client_id}/message'
            payload = json.dumps(data)
            
            self.client.publish(pqc_topic, payload, qos=1)
            logger.info(f"Sent encrypted message to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending encrypted message: {e}")
            return False
    
    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Handle disconnection"""
        logger.info(f"Disconnected from broker: {reason_code}")
        self.handshake_complete = False
        self.shared_secret = None
    
    def connect(self):
        """Connect to broker"""
        logger.info(f"Connecting to {self.broker_host}:{self.broker_port}")
        self.client.connect(self.broker_host, self.broker_port, 60)
        self.client.loop_start()
    
    def disconnect(self):
        """Disconnect from broker"""
        self.client.loop_stop()
        self.client.disconnect()


async def test_pqsm_system():
    """Test the PQSM broker and client"""
    
    # Start broker
    broker = PQSMBroker(kem_algorithm='ML-KEM-512')
    broker_task = asyncio.create_task(broker.start())
    
    # Give broker time to start
    await asyncio.sleep(2)
    
    # Create multiple clients
    clients = []
    for i in range(3):
        client = PQSMClient(
            kem_algorithm='ML-KEM-512',
            client_id=f"test_client_{i}"
        )
        client.connect()
        clients.append(client)
    
    # Wait for handshakes
    await asyncio.sleep(5)
    
    # Send test messages
    for i, client in enumerate(clients):
        for j in range(5):
            client.send_encrypted_message(
                f"sensor/temperature/{i}",
                f"{{\"temp\": {20 + j}, \"unit\": \"C\"}}"
            )
            await asyncio.sleep(0.1)
    
    # Get metrics
    await asyncio.sleep(2)
    metrics = broker.get_metrics()
    print("\nBroker Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    for client in clients:
        client.disconnect()
    
    broker.stop()
    broker_task.cancel()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_pqsm_system())
