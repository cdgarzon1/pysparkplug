"""Unit tests for Edge Node rebirth functionality

This module tests the Edge Node rebirth functionality according to the Sparkplug B specification.

Key specification requirements tested:
- NBIRTH must include Node Control/Rebirth metric
- Node Control/Rebirth must not use aliases
- Node Control/Rebirth must be Boolean with false value in NBIRTH
- Rebirth Request must use NCMD with Node Control/Rebirth=true
- Edge Node must stop DATA messages on rebirth
- Edge Node must send complete BIRTH sequence
- bdSeq must remain unchanged in new NBIRTH
"""

import logging
import random
import threading
import time
import unittest
import uuid

logger = logging.getLogger(__name__)
from typing import Any, cast
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt

from pysparkplug._client import Client
from pysparkplug._datatype import DataType
from pysparkplug._edge_node import NODE_CONTROL_REBIRTH, Device, EdgeNode
from pysparkplug._enums import MessageType, QoS
from pysparkplug._message import Message
from pysparkplug._metric import Metric
from pysparkplug._payload import NBirth, NCmd
from pysparkplug._topic import Topic

GROUP_ID = "test_group"
EDGE_NODE_ID = "test_edge_node"


class TestEdgeNodeRebirth(unittest.TestCase):
    """Test suite for Edge Node rebirth functionality"""

    def setUp(self):
        """Set up test environment before each test"""
        self.client = Client()
        self.mock_publish = MagicMock()
        self.client.publish = self.mock_publish

        # Track will message while allowing original set_will to execute
        self.original_set_will = self.client.set_will
        self.mock_set_will = MagicMock()

        def set_will_and_track(message: Message | None) -> None:
            self.mock_set_will(message)
            return self.original_set_will(message)

        self.client.set_will = set_will_and_track

        self.test_edge_metric = Metric(
            timestamp=123,
            name="test_edge_node_metric",
            datatype=DataType.INT32,
            value=42,
        )
        self.test_dev_metric = Metric(
            timestamp=123, name="test_device_metric", datatype=DataType.INT32, value=1
        )

        self.edge_node = EdgeNode(
            group_id=GROUP_ID,
            edge_node_id=EDGE_NODE_ID,
            metrics=[self.test_edge_metric],
            client=self.client,
        )

        for device_id in range(5):
            device = Device(
                device_id=f"test_device{device_id}", metrics=[self.test_dev_metric]
            )
            self.edge_node.register(device)

    def test_nbirth_rebirth_metric_requirements(self):
        """Test NBIRTH message compliance with Node Control/Rebirth requirements

        [tck-id-operational-behavior-data-commands-rebirth-name]
        [tck-id-operational-behavior-data-commands-rebirth-name-aliases]
        [tck-id-operational-behavior-data-commands-rebirth-datatype]
        [tck-id-operational-behavior-data-commands-rebirth-value]
        """
        # Track published messages to inspect initial NBIRTH
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        # Connect edge node to trigger initial NBIRTH
        self.edge_node.connect("test.mosquitto.org")
        # Get the will message that was set
        initial_will_message = self.mock_set_will.call_args[0][0]

        # Wait for connection to establish and initial NBIRTH to complete
        while self.edge_node._connected is False or len(published_msgs) < 6:
            time.sleep(0.5)

        # Find NBIRTH message
        birth_msg = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        birth_payload = cast(NBirth, birth_msg.payload)
        self.assertEqual(birth_payload.seq, 0, "NBIRTH must have seq=0")

        rebirth_metric = None
        bdseq_metric = None
        for m in birth_payload.metrics:
            if m.name == NODE_CONTROL_REBIRTH:
                rebirth_metric = m
            elif m.name == "bdSeq":
                bdseq_metric = m
        self.assertEqual(
            bdseq_metric.value,  # type: ignore[reportOptionalMemberAccess]
            initial_will_message.payload.bd_seq_metric.value,
            "bdSeq in NBIRTH must match that in the will message",
        )

        # Verify rebirth metric exists and meets requirements
        self.assertIsNotNone(
            rebirth_metric, "NBIRTH must include Node Control/Rebirth metric"
        )
        self.assertIsNone(
            rebirth_metric.alias,  # type: ignore[reportOptionalMemberAccess]
            "Node Control/Rebirth must not use aliases",
        )
        self.assertEqual(
            rebirth_metric.datatype,  # type: ignore[reportOptionalMemberAccess]
            DataType.BOOLEAN,
            "Node Control/Rebirth must be Boolean type",
        )
        self.assertFalse(
            rebirth_metric.value,  # type: ignore[reportOptionalMemberAccess]
            "Node Control/Rebirth must be False in NBIRTH",
        )

        self.edge_node.disconnect()

    def test_rebirth_command_requirements(self):
        """Test rebirth command handling according to specification

        [tck-id-operational-behavior-data-commands-ncmd-rebirth-verb]
        [tck-id-operational-behavior-data-commands-ncmd-rebirth-name]
        [tck-id-operational-behavior-data-commands-ncmd-rebirth-value]
        [tck-id-operational-behavior-data-commands-rebirth-action-1]
        [tck-id-operational-behavior-data-commands-rebirth-action-2]
        """
        # Create a valid rebirth command
        rebirth_metric = Metric(
            timestamp=123,
            name=NODE_CONTROL_REBIRTH,
            datatype=DataType.BOOLEAN,
            value=True,
        )

        ncmd_topic = Topic(
            group_id=GROUP_ID, message_type=MessageType.NCMD, edge_node_id=EDGE_NODE_ID
        )  # Correct verb

        ncmd_message = Message(
            topic=ncmd_topic,
            payload=NCmd(timestamp=123, metrics=(rebirth_metric,)),
            qos=QoS.EXACTLY_ONCE,
            retain=False,
        )

        # Track published messages
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        self.edge_node.connect("test.mosquitto.org")
        # Get the will message that was set
        initial_will_message = self.mock_set_will.call_args[0][0]

        # Wait for connection to establish and initial NBIRTH to complete
        while self.edge_node._connected is False or len(published_msgs) < 6:
            time.sleep(0.5)

        published_msgs = []  # Reset published messages

        def send_data_messages():
            """Simulate sending NDATA messages"""
            for i in range(10):
                data_metric = Metric(
                    timestamp=1234,
                    name="test_edge_node_metric",
                    datatype=DataType.INT32,
                    value=50 + i,
                )
                self.edge_node.update([data_metric])
                time.sleep(0.1)

        # simulate a user thread is sending data messages
        rebirth_thread = threading.Thread(target=send_data_messages)
        rebirth_thread.start()

        time.sleep(random.uniform(0.1, 1))  # Allow some NDATA messages to be sent
        # Send rebirth command
        self.edge_node._handle_ncmd(ncmd_message)

        rebirth_thread.join()  # Ensure data sending thread has completed

        birth_msg = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        birth_payload = cast(NBirth, birth_msg.payload)
        bd_seq_metric = next(
            (m for m in birth_payload.metrics if m.name == "bdSeq"), None
        )

        self.assertEqual(
            birth_payload.seq, 0, "NBIRTH must have seq=0 even for rebirth"
        )
        self.assertEqual(
            bd_seq_metric.value,  # type: ignore[reportOptionalMemberAccess]
            initial_will_message.payload.bd_seq_metric.value,
            "bdSeq in NBIRTH must match that in the will message even for rebirth",
        )

        # Verify message sequence
        message_types = [type(msg.payload).__name__ for msg in published_msgs]
        print(f"message_types: {message_types}")

        # Find indices of Birth and NData messages
        birth_indices = [i for i, t in enumerate(message_types) if "Birth" in t]
        ndata_indices = [i for i, t in enumerate(message_types) if t == "NData"]

        # Verify no NData messages occur between Birth messages
        if len(birth_indices) > 0:
            first_birth = birth_indices[0]
            last_birth = birth_indices[-1]
            data_between_births = [
                i for i in ndata_indices if first_birth <= i <= last_birth
            ]
            self.assertEqual(
                len(data_between_births),
                0,
                "No NData messages should occur during Birth sequence",
            )

        self.edge_node.disconnect()

    def test_rebirth_during_rebirth(self):
        """Test that rebirth commands are ignored during an ongoing rebirth"""
        # Start a rebirth
        self.edge_node._rebirth_lock.acquire()

        # Create another rebirth command
        rebirth_metric = Metric(
            timestamp=123,
            name=NODE_CONTROL_REBIRTH,
            datatype=DataType.BOOLEAN,
            value=True,
        )

        ncmd_topic = Topic(
            group_id=GROUP_ID, message_type=MessageType.NCMD, edge_node_id=EDGE_NODE_ID
        )

        ncmd_message = Message(
            topic=ncmd_topic,
            payload=NCmd(timestamp=123, metrics=(rebirth_metric,)),
            qos=QoS.EXACTLY_ONCE,
            retain=False,
        )

        # Send rebirth command
        self.edge_node._handle_ncmd(ncmd_message)

        # Verify no messages were published
        self.mock_publish.assert_not_called()
        self.edge_node._rebirth_lock.release()

    def test_invalid_rebirth_command(self):
        """Test handling of invalid rebirth commands"""
        # Test with wrong metric name
        wrong_name_metric = Metric(
            timestamp=123, name="wrong_name", datatype=DataType.BOOLEAN, value=True
        )

        # Test with wrong value type
        wrong_value_metric = Metric(
            timestamp=123,
            name=NODE_CONTROL_REBIRTH,
            datatype=DataType.BOOLEAN,
            value=False,
        )

        ncmd_topic = Topic(
            group_id=GROUP_ID, message_type=MessageType.NCMD, edge_node_id=EDGE_NODE_ID
        )

        for metric in [wrong_name_metric, wrong_value_metric]:
            message = Message(
                topic=ncmd_topic,
                payload=NCmd(timestamp=123, metrics=(metric,)),
                qos=QoS.EXACTLY_ONCE,
                retain=False,
            )

            self.edge_node._handle_ncmd(message)
            self.mock_publish.assert_not_called()

    def test_rebirth_metrics_preserved(self):
        """Test that metrics are preserved through rebirth process"""
        # Create a rebirth command
        rebirth_metric = Metric(
            timestamp=123,
            name=NODE_CONTROL_REBIRTH,
            datatype=DataType.BOOLEAN,
            value=True,
        )

        ncmd_topic = Topic(
            group_id=GROUP_ID, message_type=MessageType.NCMD, edge_node_id=EDGE_NODE_ID
        )

        ncmd_message = Message(
            topic=ncmd_topic,
            payload=NCmd(timestamp=123, metrics=(rebirth_metric,)),
            qos=QoS.EXACTLY_ONCE,
            retain=False,
        )

        # Track published messages
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        self.edge_node.connect("test.mosquitto.org")

        # Wait for connection to establish and initial NBIRTH to complete
        while self.edge_node._connected is False or len(published_msgs) < 6:
            time.sleep(0.5)

        published_msgs = []  # Reset published messages

        # Send rebirth command
        self.edge_node._handle_ncmd(ncmd_message)

        while len(published_msgs) < 6:  # wait for all birth messages
            time.sleep(0.1)

        # Find NBirth message
        birth_msg = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )

        # Verify original metrics are present in birth message
        birth_payload = birth_msg.payload
        assert isinstance(birth_payload, NBirth)  # type narrowing
        birth_metrics = {m.name: m.value for m in birth_payload.metrics}
        self.assertIn("test_edge_node_metric", birth_metrics)
        self.assertEqual(birth_metrics["test_edge_node_metric"], 42)
        self.edge_node.disconnect()

    def test_rebirth_bdseq_handling(self):
        """Test birth/death sequence number handling during rebirth

        [tck-id-operational-behavior-data-commands-rebirth-action-3]
        """
        # Track published messages to capture initial MQTT CONNECT will message
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        # Connect to capture initial will message bdSeq
        self.edge_node.connect("test.mosquitto.org")
        while self.edge_node._connected is False or len(published_msgs) < 6:
            time.sleep(0.5)

        initial_birth = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        initial_birth_payload = cast(NBirth, initial_birth.payload)
        initial_bdseq = next(
            m.value for m in initial_birth_payload.metrics if m.name == "bdSeq"
        )
        published_msgs.clear()  # Reset for rebirth test

        # Create and send rebirth command
        rebirth_metric = Metric(
            timestamp=123,
            name=NODE_CONTROL_REBIRTH,
            datatype=DataType.BOOLEAN,
            value=True,
        )

        ncmd_topic = Topic(
            group_id=GROUP_ID, message_type=MessageType.NCMD, edge_node_id=EDGE_NODE_ID
        )

        ncmd_message = Message(
            topic=ncmd_topic,
            payload=NCmd(timestamp=123, metrics=(rebirth_metric,)),
            qos=QoS.EXACTLY_ONCE,
            retain=False,
        )

        # Send rebirth command
        self.edge_node._handle_ncmd(ncmd_message)
        while len(published_msgs) < 6:  # wait for all birth messages
            time.sleep(0.1)

        # Find Birth message after rebirth
        rebirth_msg = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        rebirth_payload = cast(NBirth, rebirth_msg.payload)

        # Get bdSeq after rebirth
        rebirth_bdseq = next(
            m.value for m in rebirth_payload.metrics if m.name == "bdSeq"
        )

        # Verify bdSeq hasn't changed

        self.assertEqual(
            rebirth_bdseq,
            initial_bdseq,
            "bdSeq must not change during rebirth (no new MQTT session)",
        )
        self.edge_node.disconnect()

    def test_bdseq_increment_after_unexpected_disconnect(self):
        """Test that bdSeq is incremented after unexpected disconnection and auto-reconnect"""
        # Track published messages
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        # Now connect the edge node
        self.edge_node.connect("test.mosquitto.org")

        # Wait for connection and initial NBIRTH
        while (
            self.edge_node._connected is False or len(published_msgs) < 6
        ):  # Wait for initial NBIRTH + 5 DBIRTH
            time.sleep(0.5)

        # Get initial bdSeq
        initial_birth = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        initial_birth_payload = cast(NBirth, initial_birth.payload)
        initial_bdseq = next(
            m.value for m in initial_birth_payload.metrics if m.name == "bdSeq"
        )

        published_msgs.clear()  # Reset messages

        # Simulate an unexpected disconnection by directly calling the disconnect callback
        self.client._client.on_disconnect(self.client._client, None, 1)  # type: ignore[misc]
        self.client._client.on_connect(self.client._client, None, None, 0)  # type: ignore[misc]

        # Wait for auto-reconnect and new NBIRTH
        retries = 0
        max_retries = 10
        while (
            len(published_msgs) < 6 and retries < max_retries
        ):  # Wait for reconnect NBIRTH + 5 DBIRTH
            time.sleep(1)
            retries += 1

        self.assertLess(retries, max_retries, "Failed to receive reconnection messages")

        # Get new bdSeq
        new_birth = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        new_birth_payload = cast(NBirth, new_birth.payload)
        new_bdseq = next(
            m.value for m in new_birth_payload.metrics if m.name == "bdSeq"
        )

        # Verify bdSeq was incremented
        self.assertEqual(
            new_bdseq,
            initial_bdseq + 1,  # type: ignore[reportOperatorIssue]
            "bdSeq must increment by 1 after unexpected disconnect/reconnect",
        )

        self.edge_node.disconnect()

    def test_bdseq_increment_after_disconnect(self):
        """Test that bdSeq is incremented after disconnect and reconnect"""
        # Track published messages
        published_msgs: list[Message] = []

        def track_publish(msg: Message, **kwargs: Any) -> None:
            published_msgs.append(msg)

        self.mock_publish.side_effect = track_publish

        # Connect first time
        self.edge_node.connect("test.mosquitto.org")

        # Wait for connection and initial NBIRTH
        while (
            self.edge_node._connected is False or len(published_msgs) < 6
        ):  # Wait for initial NBIRTH + 5 DBIRTH
            time.sleep(0.5)

        # Get initial bdSeq
        initial_birth = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        initial_birth_payload = cast(NBirth, initial_birth.payload)
        initial_bdseq = next(
            m.value for m in initial_birth_payload.metrics if m.name == "bdSeq"
        )

        # Disconnect
        self.edge_node.disconnect()
        time.sleep(1)  # Wait for disconnect to complete

        published_msgs.clear()  # Reset messages

        # Reconnect
        self.edge_node.connect("test.mosquitto.org")

        # Wait for new connection and NBIRTH
        while self.edge_node._connected is False or len(published_msgs) < 6:
            time.sleep(0.5)

        # Get new bdSeq
        new_birth = next(
            msg for msg in published_msgs if isinstance(msg.payload, NBirth)
        )
        new_birth_payload = cast(NBirth, new_birth.payload)
        new_bdseq = next(
            m.value for m in new_birth_payload.metrics if m.name == "bdSeq"
        )

        # Verify bdSeq was incremented
        self.assertEqual(
            new_bdseq,
            initial_bdseq + 1,  # type: ignore[reportOptionalMemberAccess]
            "bdSeq must increment by 1 after disconnect/reconnect",
        )

        self.edge_node.disconnect()


class TestEdgeNodeRebirthWithRealMQTT(unittest.TestCase):
    """Test suite for Edge Node rebirth functionality using real MQTT messages"""

    def setUp(self):
        """Set up test environment before each test"""

        self.group_id = f"test_group_{uuid.uuid4().hex[:32]}"  # Unique group ID
        self.edge_node_id = "test_edge_node"

        # Create MQTT client for sending commands and receiving messages
        self.mqtt_client = mqtt.Client()
        self.received_messages = []

        def on_message(
            client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage
        ) -> None:
            self.received_messages.append(message)
            logger.info("Test Receiver: Received message on topic %s", message.topic)

        self.mqtt_client.on_message = on_message
        self.mqtt_client.connect("test.mosquitto.org", 1883, 60)

        # Subscribe to all relevant topics
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/NBIRTH/#", qos=0)
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/DBIRTH/#", qos=0)
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/NDATA/#", qos=0)
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/DDATA/#", qos=0)
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/NDEATH/#", qos=0)
        self.mqtt_client.subscribe(f"spBv1.0/{self.group_id}/DDEATH/#", qos=0)

        self.mqtt_client.loop_start()

        # Create Edge Node with metrics
        self.client = Client()
        self.test_edge_metric = Metric(
            timestamp=123,
            name="test_edge_node_metric",
            datatype=DataType.INT32,
            value=42,
        )
        self.edge_node = EdgeNode(
            group_id=self.group_id,
            edge_node_id=self.edge_node_id,
            metrics=[self.test_edge_metric],
            client=self.client,
        )

        # Create and register 5 devices
        for i in range(5):
            device = Device(
                device_id=f"device_{i}",
                metrics=[
                    Metric(
                        timestamp=123,
                        name=f"device_{i}_metric",
                        datatype=DataType.INT32,
                        value=i,
                    )
                ],
            )
            self.edge_node.register(device)

    def tearDown(self):
        """Clean up after each test"""
        self.edge_node.disconnect()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        time.sleep(1)  # Allow time for disconnection

    def test_rebirth_flow_with_real_mqtt(self):
        """Test complete rebirth flow using real MQTT messages"""
        # Connect edge node and wait for initial birth sequence
        logger.info("Connecting edge node...")
        self.edge_node.connect("test.mosquitto.org", keepalive=120)

        # Wait for initial birth sequence to complete and ensure we're stable
        initial_msg_count = 0
        retry_count = 0
        max_retries = 10

        while retry_count < max_retries:
            time.sleep(0.5)  # Longer delay for stability
            current_count = len(self.received_messages)
            logger.info(
                "Current message count: %d, Previous count: %d",
                current_count,
                initial_msg_count,
            )

            if current_count >= 6 and current_count == initial_msg_count:
                logger.info("Message count has stabilized at %d", current_count)
                break  # Message count has stabilized

            initial_msg_count = current_count
            retry_count += 1

        # Clear all messages after initial birth sequence
        logger.info("Clearing %d initial messages", len(self.received_messages))
        self.received_messages = []
        time.sleep(1.0)  # Longer stability delay

        # Create and send rebirth command using proper Sparkplug B encoding
        ncmd_topic = Topic(
            group_id=self.group_id,
            message_type=MessageType.NCMD,
            edge_node_id=self.edge_node_id,
        )
        rebirth_metric = Metric(
            timestamp=int(time.time() * 1000),
            name="Node Control/Rebirth",
            datatype=DataType.BOOLEAN,
            value=True,
        )
        ncmd_payload = NCmd(
            timestamp=int(time.time() * 1000), metrics=(rebirth_metric,)
        )
        message = Message(
            topic=ncmd_topic, payload=ncmd_payload, qos=QoS.EXACTLY_ONCE, retain=False
        )

        self.mqtt_client.publish(
            topic=str(message.topic),
            payload=message.payload.encode(include_dtypes=True),
            qos=message.qos.value,
            retain=message.retain,
        )

        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            time.sleep(0.5)
            current_count = len(self.received_messages)
            logger.info(
                "Waiting for rebirth messages. Current count: %d", current_count
            )

            if current_count >= 6:  # We have our expected messages
                break

            retry_count += 1

        logger.info(
            "Processing %d messages after rebirth command", len(self.received_messages)
        )
        proc_msgs = []
        nbirth = 0
        dbirth = 0
        for msg in self.received_messages:
            proc_msg = self.client._handle_message(msg)
            proc_msgs.append(proc_msg)
            if "NBIRTH" in str(proc_msg.topic):
                nbirth += 1
                logger.debug("Found NBIRTH message: %s", proc_msg)
            elif "DBIRTH" in str(proc_msg.topic):
                dbirth += 1
                logger.debug("Found DBIRTH message: %s", proc_msg)
        self.assertEqual(nbirth, 1)
        self.assertEqual(dbirth, 5)
