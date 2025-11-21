#!/usr/bin/env python3
"""
Network Simulator for PQSM Research
Simulates various network conditions for IoT/MQTT testing
Author: Varanasi Sai Srinivasa Karthik
"""

import simpy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import time
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkCondition(Enum):
    """Network condition profiles"""
    IDEAL = "ideal"
    LAN = "lan"
    WIFI = "wifi"
    CELLULAR_4G = "4g"
    CELLULAR_3G = "3g"
    SATELLITE = "satellite"
    CONGESTED = "congested"


@dataclass
class NetworkProfile:
    """Network characteristics"""
    name: str
    latency_ms: float
    jitter_ms: float
    packet_loss: float
    bandwidth_mbps: float
    
    def get_actual_latency(self) -> float:
        """Get latency with jitter"""
        return max(0, self.latency_ms + random.gauss(0, self.jitter_ms))
    
    def should_drop_packet(self) -> bool:
        """Determine if packet should be dropped"""
        return random.random() < self.packet_loss


# Define network profiles
NETWORK_PROFILES = {
    NetworkCondition.IDEAL: NetworkProfile("ideal", 0.5, 0.1, 0.0, 1000),
    NetworkCondition.LAN: NetworkProfile("lan", 1.0, 0.2, 0.001, 1000),
    NetworkCondition.WIFI: NetworkProfile("wifi", 5.0, 2.0, 0.01, 100),
    NetworkCondition.CELLULAR_4G: NetworkProfile("4g", 30.0, 10.0, 0.02, 50),
    NetworkCondition.CELLULAR_3G: NetworkProfile("3g", 100.0, 30.0, 0.05, 10),
    NetworkCondition.SATELLITE: NetworkProfile("satellite", 600.0, 50.0, 0.03, 25),
    NetworkCondition.CONGESTED: NetworkProfile("congested", 150.0, 100.0, 0.08, 5)
}


@dataclass
class Message:
    """Network message"""
    msg_id: str
    sender: str
    receiver: str
    payload_bytes: int
    kem_algorithm: str
    timestamp: float
    priority: int = 0
    
    def __hash__(self):
        return hash(self.msg_id)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    total_latency_ms: float = 0.0
    handshakes_completed: int = 0
    handshake_failures: int = 0
    total_bytes: int = 0
    simulation_time: float = 0.0
    
    # Per-KEM metrics
    kem_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_message(self, msg: Message, delivered: bool, latency_ms: float):
        """Record message metrics"""
        self.messages_sent += 1
        if delivered:
            self.messages_received += 1
            self.total_latency_ms += latency_ms
        else:
            self.messages_dropped += 1
        
        self.total_bytes += msg.payload_bytes
        
        # Update per-KEM metrics
        if msg.kem_algorithm not in self.kem_metrics:
            self.kem_metrics[msg.kem_algorithm] = {
                'sent': 0, 'received': 0, 'dropped': 0,
                'total_latency': 0.0, 'latencies': []
            }
        
        kem_stats = self.kem_metrics[msg.kem_algorithm]
        kem_stats['sent'] += 1
        if delivered:
            kem_stats['received'] += 1
            kem_stats['total_latency'] += latency_ms
            kem_stats['latencies'].append(latency_ms)
        else:
            kem_stats['dropped'] += 1
    
    def get_delivery_ratio(self) -> float:
        """Calculate delivery ratio"""
        if self.messages_sent == 0:
            return 0.0
        return self.messages_received / self.messages_sent
    
    def get_avg_latency(self) -> float:
        """Calculate average latency"""
        if self.messages_received == 0:
            return 0.0
        return self.total_latency_ms / self.messages_received
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary = {
            'total_messages': self.messages_sent,
            'delivered': self.messages_received,
            'dropped': self.messages_dropped,
            'delivery_ratio': self.get_delivery_ratio(),
            'avg_latency_ms': self.get_avg_latency(),
            'throughput_mbps': (self.total_bytes * 8) / (self.simulation_time * 1_000_000),
            'handshakes_completed': self.handshakes_completed,
            'handshake_success_rate': self.handshakes_completed / 
                                     (self.handshakes_completed + self.handshake_failures)
                                     if (self.handshakes_completed + self.handshake_failures) > 0 else 0
        }
        
        # Add per-KEM statistics
        for kem, stats in self.kem_metrics.items():
            if stats['received'] > 0:
                summary[f'{kem}_delivery_ratio'] = stats['received'] / stats['sent']
                summary[f'{kem}_avg_latency'] = stats['total_latency'] / stats['received']
                summary[f'{kem}_p95_latency'] = np.percentile(stats['latencies'], 95) if stats['latencies'] else 0
        
        return summary


class NetworkSimulator:
    """Main network simulator"""
    
    def __init__(self, env: simpy.Environment, profile: NetworkProfile):
        self.env = env
        self.profile = profile
        self.metrics = SimulationMetrics()
        self.message_queue = simpy.Store(env)
        self.nodes = {}
        
    def add_node(self, node_id: str, node_type: str = "device"):
        """Add a network node"""
        self.nodes[node_id] = {
            'type': node_type,
            'connected': True,
            'messages_sent': 0,
            'messages_received': 0
        }
    
    def send_message(self, msg: Message):
        """Send a message through the network"""
        # Check if nodes exist
        if msg.sender not in self.nodes or msg.receiver not in self.nodes:
            logger.warning(f"Invalid nodes: {msg.sender} -> {msg.receiver}")
            return
        
        # Check if packet should be dropped
        if self.profile.should_drop_packet():
            self.metrics.add_message(msg, False, 0)
            logger.debug(f"Packet dropped: {msg.msg_id}")
            return
        
        # Calculate transmission delay
        latency_ms = self.profile.get_actual_latency()
        
        # Add bandwidth delay
        bandwidth_delay = (msg.payload_bytes * 8) / (self.profile.bandwidth_mbps * 1_000_000) * 1000
        total_delay = latency_ms + bandwidth_delay
        
        # Schedule delivery
        self.env.process(self._deliver_message(msg, total_delay))
    
    def _deliver_message(self, msg: Message, delay_ms: float):
        """Deliver message after delay"""
        yield self.env.timeout(delay_ms / 1000)  # Convert to seconds
        
        # Record successful delivery
        self.metrics.add_message(msg, True, delay_ms)
        self.nodes[msg.receiver]['messages_received'] += 1
        
        logger.debug(f"Message delivered: {msg.msg_id} ({delay_ms:.2f}ms)")
    
    def perform_handshake(self, client_id: str, server_id: str, kem_algorithm: str):
        """Simulate KEM handshake"""
        # KEM-specific handshake times (based on our measurements)
        handshake_times = {
            'ML-KEM-512': 21.9,
            'ML-KEM-768': 32.8,
            'ML-KEM-1024': 47.3,
            'NTRU-Prime-hrss': 26.1,
            'BIKE-L1': 50.5,
            'HQC-128': 55.1,
            'Classic-McEliece-348864': 278.3
        }
        
        base_time = handshake_times.get(kem_algorithm, 30.0)
        
        # Add network latency (3 round trips for handshake)
        network_time = 3 * self.profile.get_actual_latency()
        total_time = base_time + network_time
        
        # Check for handshake failure
        failure_prob = self.profile.packet_loss * 3  # Three chances to fail
        if random.random() < failure_prob:
            self.metrics.handshake_failures += 1
            logger.debug(f"Handshake failed: {client_id} <-> {server_id}")
            return False
        
        # Simulate handshake delay
        yield self.env.timeout(total_time / 1000)
        
        self.metrics.handshakes_completed += 1
        logger.debug(f"Handshake completed: {client_id} <-> {server_id} ({total_time:.2f}ms)")
        return True


class IoTDevice:
    """Simulated IoT device"""
    
    def __init__(self, device_id: str, simulator: NetworkSimulator, 
                 kem_algorithm: str = 'ML-KEM-512',
                 message_rate_hz: float = 1.0,
                 payload_size: int = 256):
        self.device_id = device_id
        self.simulator = simulator
        self.kem_algorithm = kem_algorithm
        self.message_rate_hz = message_rate_hz
        self.payload_size = payload_size
        self.connected = False
        self.message_count = 0
    
    def connect(self, broker_id: str = 'broker'):
        """Connect to broker with PQC handshake"""
        logger.info(f"Device {self.device_id} connecting with {self.kem_algorithm}")
        
        # Perform handshake
        success = yield from self.simulator.perform_handshake(
            self.device_id, broker_id, self.kem_algorithm
        )
        
        self.connected = success
        return success
    
    def send_telemetry(self, broker_id: str = 'broker'):
        """Send telemetry messages"""
        if not self.connected:
            yield self.simulator.env.timeout(1)
            return
        
        while True:
            # Create message
            msg = Message(
                msg_id=f"{self.device_id}_msg_{self.message_count}",
                sender=self.device_id,
                receiver=broker_id,
                payload_bytes=self.payload_size,
                kem_algorithm=self.kem_algorithm,
                timestamp=self.simulator.env.now
            )
            
            # Send message
            self.simulator.send_message(msg)
            self.message_count += 1
            
            # Wait for next message
            interval = 1.0 / self.message_rate_hz
            yield self.simulator.env.timeout(interval)


def run_simulation(duration_seconds: int = 30,
                  num_devices: int = 10,
                  network_condition: NetworkCondition = NetworkCondition.WIFI,
                  kem_algorithms: List[str] = None) -> SimulationMetrics:
    """Run a network simulation"""
    
    if kem_algorithms is None:
        kem_algorithms = ['ML-KEM-512', 'NTRU-Prime-hrss', 'BIKE-L1']
    
    # Create environment
    env = simpy.Environment()
    
    # Create network simulator
    profile = NETWORK_PROFILES[network_condition]
    simulator = NetworkSimulator(env, profile)
    
    # Add broker
    simulator.add_node('broker', 'broker')
    
    # Create devices
    devices = []
    for i in range(num_devices):
        # Distribute KEM algorithms
        kem = kem_algorithms[i % len(kem_algorithms)]
        
        device = IoTDevice(
            device_id=f"device_{i}",
            simulator=simulator,
            kem_algorithm=kem,
            message_rate_hz=random.uniform(0.5, 2.0),
            payload_size=random.choice([128, 256, 512, 1024])
        )
        
        simulator.add_node(device.device_id, 'device')
        devices.append(device)
        
        # Schedule device operations
        env.process(device_lifecycle(env, device, simulator))
    
    # Run simulation
    start_time = time.time()
    env.run(until=duration_seconds)
    
    # Update metrics
    simulator.metrics.simulation_time = time.time() - start_time
    
    return simulator.metrics


def device_lifecycle(env: simpy.Environment, device: IoTDevice, simulator: NetworkSimulator):
    """Device lifecycle process"""
    # Connect to broker
    yield from device.connect()
    
    if device.connected:
        # Start sending telemetry
        yield from device.send_telemetry()


def run_comprehensive_test():
    """Run comprehensive network simulation test"""
    
    results = []
    
    # Test different network conditions
    for condition in NetworkCondition:
        logger.info(f"\nTesting {condition.value} network...")
        
        metrics = run_simulation(
            duration_seconds=30,
            num_devices=20,
            network_condition=condition,
            kem_algorithms=['ML-KEM-512', 'NTRU-Prime-hrss', 'BIKE-L1', 
                          'HQC-128', 'Classic-McEliece-348864']
        )
        
        summary = metrics.get_summary()
        summary['network_condition'] = condition.value
        results.append(summary)
        
        # Print results
        print(f"\n{condition.value.upper()} Network Results:")
        print(f"  Delivery Ratio: {summary['delivery_ratio']:.3f}")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {summary['throughput_mbps']:.2f} Mbps")
        print(f"  Handshake Success: {summary['handshake_success_rate']:.3f}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('network_simulation_results.csv', index=False)
    
    print("\nResults saved to network_simulation_results.csv")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("PQSM Network Simulator")
    print("="*60)
    
    # Run comprehensive test
    results = run_comprehensive_test()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Calculate averages
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\nAverage Performance Across All Conditions:")
    print(f"  Overall Delivery Ratio: {df['delivery_ratio'].mean():.3f}")
    print(f"  Overall Avg Latency: {df['avg_latency_ms'].mean():.2f}ms")
    print(f"  Overall Throughput: {df['throughput_mbps'].mean():.2f} Mbps")
