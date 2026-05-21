"""Comprehensive unit tests for PropertyValue, PropertySet, and PropertySetList

Tests are organized by spec requirements:

- PropertyValue: type field required; valid datatypes; null handling; scalar encoding;
  nested PropertySet and PropertySetList values.
- PropertySet: parallel keys/values; length validation; empty; roundtrip encoding.
- PropertySetList: ordered collection; roundtrip encoding.
- Metric.properties: optional field; absent when None; roundtrip with complex structures.
- QualityCode: BAD/GOOD/STALE values; MUST use INT32 datatype; MUST use int_value field.
"""

import datetime
import unittest

from pysparkplug._datatype import DataType
from pysparkplug._metric import Metric
from pysparkplug._properties import (
    QUALITY_KEY,
    PropertySet,
    PropertySetList,
    PropertyValue,
    QualityCode,
)


class TestPropertyValueScalars(unittest.TestCase):
    """Test PropertyValue with all supported scalar DataTypes."""

    def _roundtrip(
        self,
        datatype: DataType,
        value: "int | float | bool | str | datetime.datetime | PropertySet | PropertySetList | None",
    ) -> PropertyValue:
        """Helper: PropertyValue → protobuf → PropertyValue."""
        pv = PropertyValue(datatype=datatype, value=value)
        pb = pv.to_pb()
        return PropertyValue.from_pb(pb)

    def test_type_field_always_serialized(self):
        """The protobuf 'type' field MUST always be present in encoded form."""
        for dt in (DataType.INT32, DataType.BOOLEAN, DataType.STRING, DataType.FLOAT):
            with self.subTest(datatype=dt):
                pv = PropertyValue(datatype=dt, value=None, is_null=True)
                pb = pv.to_pb()
                self.assertEqual(pb.type, dt)

    def test_int8(self):
        pv2 = self._roundtrip(DataType.INT8, -128)
        self.assertEqual(pv2.value, -128)
        self.assertEqual(pv2.datatype, DataType.INT8)

        pv2 = self._roundtrip(DataType.INT8, 127)
        self.assertEqual(pv2.value, 127)

    def test_int8_overflow(self):
        with self.assertRaises(OverflowError):
            PropertyValue(datatype=DataType.INT8, value=128).to_pb()
        with self.assertRaises(OverflowError):
            PropertyValue(datatype=DataType.INT8, value=-129).to_pb()

    def test_int16(self):
        pv2 = self._roundtrip(DataType.INT16, -32768)
        self.assertEqual(pv2.value, -32768)
        pv2 = self._roundtrip(DataType.INT16, 32767)
        self.assertEqual(pv2.value, 32767)

    def test_int32(self):
        pv2 = self._roundtrip(DataType.INT32, -2147483648)
        self.assertEqual(pv2.value, -2147483648)
        pv2 = self._roundtrip(DataType.INT32, 2147483647)
        self.assertEqual(pv2.value, 2147483647)

    def test_int64(self):
        pv2 = self._roundtrip(DataType.INT64, -(2**63))
        self.assertEqual(pv2.value, -(2**63))
        pv2 = self._roundtrip(DataType.INT64, 2**63 - 1)
        self.assertEqual(pv2.value, 2**63 - 1)

    def test_uint8(self):
        pv2 = self._roundtrip(DataType.UINT8, 0)
        self.assertEqual(pv2.value, 0)
        pv2 = self._roundtrip(DataType.UINT8, 255)
        self.assertEqual(pv2.value, 255)

    def test_uint16(self):
        pv2 = self._roundtrip(DataType.UINT16, 65535)
        self.assertEqual(pv2.value, 65535)

    def test_uint32(self):
        pv2 = self._roundtrip(DataType.UINT32, 0xFFFFFFFF)
        self.assertEqual(pv2.value, 0xFFFFFFFF)

    def test_uint64(self):
        pv2 = self._roundtrip(DataType.UINT64, 2**64 - 1)
        self.assertEqual(pv2.value, 2**64 - 1)

    def test_float(self):
        import struct

        raw_value = 3.14
        # Float has limited precision; compare encoded roundtrip
        pv2 = self._roundtrip(DataType.FLOAT, raw_value)
        expected = struct.unpack("f", struct.pack("f", raw_value))[0]
        self.assertAlmostEqual(pv2.value, expected, places=5)  # float precision

    def test_double(self):
        pv2 = self._roundtrip(DataType.DOUBLE, 3.141592653589793)
        self.assertAlmostEqual(pv2.value, 3.141592653589793, places=12)  # type: ignore[arg-type]

    def test_boolean_true(self):
        pv2 = self._roundtrip(DataType.BOOLEAN, True)
        self.assertIs(pv2.value, True)

    def test_boolean_false(self):
        pv2 = self._roundtrip(DataType.BOOLEAN, False)
        self.assertIs(pv2.value, False)

    def test_string(self):
        pv2 = self._roundtrip(DataType.STRING, "hello world")
        self.assertEqual(pv2.value, "hello world")

    def test_string_empty(self):
        pv2 = self._roundtrip(DataType.STRING, "")
        self.assertEqual(pv2.value, "")

    def test_string_unicode(self):
        pv2 = self._roundtrip(DataType.STRING, "こんにちは 🌍")
        self.assertEqual(pv2.value, "こんにちは 🌍")

    def test_datetime(self):
        ts = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        pv2 = self._roundtrip(DataType.DATETIME, ts)
        # Datetime is encoded as ms since epoch; compare to ms precision
        assert isinstance(pv2.value, datetime.datetime)
        self.assertEqual(int(pv2.value.timestamp() * 1000), int(ts.timestamp() * 1000))

    def test_text(self):
        pv2 = self._roundtrip(DataType.TEXT, "some long text value")
        self.assertEqual(pv2.value, "some long text value")

    def test_uuid(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        pv2 = self._roundtrip(DataType.UUID, uuid_str)
        self.assertEqual(pv2.value, uuid_str)

    def test_int_value_protobuf_field(self):
        """INT8-INT32 and UINT8-UINT32 must use the protobuf 'int_value' field."""
        for dt, val in [
            (DataType.INT8, 10),
            (DataType.INT16, 1000),
            (DataType.INT32, 100000),
            (DataType.UINT8, 200),
            (DataType.UINT16, 50000),
            (DataType.UINT32, 4000000000),
        ]:
            with self.subTest(datatype=dt):
                pb = PropertyValue(datatype=dt, value=val).to_pb()
                self.assertEqual(pb.WhichOneof("value"), "int_value")

    def test_long_value_protobuf_field(self):
        """INT64, UINT64, DATETIME must use the protobuf 'long_value' field."""
        now = datetime.datetime.now(datetime.timezone.utc)
        for dt, val in [
            (DataType.INT64, 2**40),
            (DataType.UINT64, 2**60),
            (DataType.DATETIME, now),
        ]:
            with self.subTest(datatype=dt):
                pb = PropertyValue(datatype=dt, value=val).to_pb()
                self.assertEqual(pb.WhichOneof("value"), "long_value")

    def test_string_value_protobuf_field(self):
        """STRING, TEXT, UUID must use the protobuf 'string_value' field."""
        for dt, val in [
            (DataType.STRING, "abc"),
            (DataType.TEXT, "text"),
            (DataType.UUID, "550e8400-e29b-41d4-a716-446655440000"),
        ]:
            with self.subTest(datatype=dt):
                pb = PropertyValue(datatype=dt, value=val).to_pb()
                self.assertEqual(pb.WhichOneof("value"), "string_value")


class TestPropertyValueNullHandling(unittest.TestCase):
    """Test PropertyValue null semantics."""

    def test_explicit_is_null_sets_is_null_flag(self):
        """is_null=True must encode is_null=True in protobuf."""
        pv = PropertyValue(datatype=DataType.INT32, is_null=True)
        pb = pv.to_pb()
        self.assertTrue(pb.is_null)
        # No value field should be set
        self.assertIsNone(pb.WhichOneof("value"))

    def test_none_value_sets_is_null_flag(self):
        """value=None (without is_null) should also encode as null."""
        pv = PropertyValue(datatype=DataType.STRING, value=None)
        pb = pv.to_pb()
        self.assertTrue(pb.is_null)

    def test_null_roundtrip_via_is_null(self):
        """Explicit null must round-trip correctly."""
        pv = PropertyValue(datatype=DataType.FLOAT, is_null=True)
        pb = pv.to_pb()
        pv2 = PropertyValue.from_pb(pb)
        self.assertEqual(pv2.datatype, DataType.FLOAT)
        self.assertTrue(pv2.is_null)
        self.assertIsNone(pv2.value)

    def test_null_roundtrip_via_none_value(self):
        """None value must round-trip with is_null=True in decoded form."""
        pv = PropertyValue(datatype=DataType.BOOLEAN, value=None)
        pb = pv.to_pb()
        pv2 = PropertyValue.from_pb(pb)
        self.assertTrue(pv2.is_null)
        self.assertIsNone(pv2.value)

    def test_null_for_all_scalar_types(self):
        """Null must work for every valid scalar datatype."""
        for dt in (
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.BOOLEAN,
            DataType.STRING,
            DataType.DATETIME,
            DataType.TEXT,
            DataType.UUID,
        ):
            with self.subTest(datatype=dt):
                pv = PropertyValue(datatype=dt, value=None)
                pb = pv.to_pb()
                self.assertTrue(pb.is_null)
                pv2 = PropertyValue.from_pb(pb)
                self.assertIsNone(pv2.value)
                self.assertTrue(pv2.is_null)

    def test_type_field_set_even_when_null(self):
        """The type field must still be set even when is_null=True."""
        pv = PropertyValue(datatype=DataType.DOUBLE, is_null=True)
        pb = pv.to_pb()
        self.assertEqual(pb.type, DataType.DOUBLE)
        self.assertTrue(pb.is_null)


class TestPropertySetBasic(unittest.TestCase):
    """Test PropertySet structure and operations."""

    def test_empty_propertyset(self):
        """An empty PropertySet (no keys/values) must serialize correctly."""
        ps = PropertySet(keys=(), values=())
        pb = ps.to_pb()
        self.assertEqual(len(pb.keys), 0)
        self.assertEqual(len(pb.values), 0)

        ps2 = PropertySet.from_pb(pb)
        self.assertEqual(ps2.keys, ())
        self.assertEqual(ps2.values, ())

    def test_single_int32_property(self):
        """A single INT32 property must round-trip correctly."""
        ps = PropertySet(
            keys=("temperature",),
            values=(PropertyValue(datatype=DataType.INT32, value=25),),
        )
        pb = ps.to_pb()
        self.assertEqual(list(pb.keys), ["temperature"])
        self.assertEqual(pb.values[0].int_value, 25)
        self.assertEqual(pb.values[0].type, DataType.INT32)

        ps2 = PropertySet.from_pb(pb)
        self.assertEqual(ps2.keys, ("temperature",))
        self.assertEqual(ps2.values[0].value, 25)
        self.assertEqual(ps2.values[0].datatype, DataType.INT32)

    def test_multiple_mixed_types(self):
        """PropertySet with multiple different DataTypes must round-trip."""
        ps = PropertySet(
            keys=("count", "label", "active", "ratio"),
            values=(
                PropertyValue(datatype=DataType.UINT32, value=42),
                PropertyValue(datatype=DataType.STRING, value="sensor_1"),
                PropertyValue(datatype=DataType.BOOLEAN, value=True),
                PropertyValue(datatype=DataType.DOUBLE, value=0.987),
            ),
        )
        pb = ps.to_pb()
        self.assertEqual(len(pb.keys), 4)
        self.assertEqual(len(pb.values), 4)

        ps2 = PropertySet.from_pb(pb)
        self.assertEqual(ps2.keys, ("count", "label", "active", "ratio"))
        self.assertEqual(ps2.values[0].value, 42)
        self.assertEqual(ps2.values[1].value, "sensor_1")
        self.assertIs(ps2.values[2].value, True)
        self.assertAlmostEqual(ps2.values[3].value, 0.987, places=6)  # type: ignore[arg-type]

    def test_keys_values_length_mismatch_raises(self):
        """Mismatched key/value lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            PropertySet(
                keys=("a", "b"),
                values=(PropertyValue(datatype=DataType.INT32, value=1),),
            )
        with self.assertRaises(ValueError):
            PropertySet(
                keys=("a",),
                values=(
                    PropertyValue(datatype=DataType.INT32, value=1),
                    PropertyValue(datatype=DataType.INT32, value=2),
                ),
            )

    def test_keys_values_parallel_ordering(self):
        """Keys and values must maintain their order through roundtrip."""
        keys = tuple(f"key_{i}" for i in range(10))
        values = tuple(
            PropertyValue(datatype=DataType.INT32, value=i) for i in range(10)
        )
        ps = PropertySet(keys=keys, values=values)
        ps2 = PropertySet.from_pb(ps.to_pb())
        self.assertEqual(ps2.keys, keys)
        for i, v in enumerate(ps2.values):
            self.assertEqual(v.value, i)

    def test_frozen_dataclass(self):
        """PropertySet must be immutable (frozen dataclass)."""
        ps = PropertySet(
            keys=("x",),
            values=(PropertyValue(datatype=DataType.INT32, value=1),),
        )
        with self.assertRaises((AttributeError, TypeError)):
            ps.keys = ("y",)  # type: ignore[misc]

    def test_protobuf_keys_length_equals_values_length(self):
        """After serialization, len(pb.keys) must equal len(pb.values)."""
        ps = PropertySet(
            keys=("a", "b", "c"),
            values=(
                PropertyValue(datatype=DataType.INT32, value=1),
                PropertyValue(datatype=DataType.BOOLEAN, value=False),
                PropertyValue(datatype=DataType.STRING, value="test"),
            ),
        )
        pb = ps.to_pb()
        self.assertEqual(len(pb.keys), len(pb.values))

    def test_all_null_values(self):
        """A PropertySet where every value is null must serialize correctly."""
        ps = PropertySet(
            keys=("a", "b"),
            values=(
                PropertyValue(datatype=DataType.INT32, is_null=True),
                PropertyValue(datatype=DataType.STRING, value=None),
            ),
        )
        pb = ps.to_pb()
        ps2 = PropertySet.from_pb(pb)
        for v in ps2.values:
            self.assertTrue(v.is_null)
            self.assertIsNone(v.value)


class TestPropertySetNested(unittest.TestCase):
    """Test nested PropertySet (PROPERTYSET datatype) within a PropertySet."""

    def test_propertyset_in_propertyset(self):
        """A PropertyValue with datatype=PROPERTYSET must hold a nested PropertySet."""
        inner = PropertySet(
            keys=("unit",),
            values=(PropertyValue(datatype=DataType.STRING, value="degC"),),
        )
        outer = PropertySet(
            keys=("temperature",),
            values=(
                PropertyValue(
                    datatype=DataType.PROPERTYSET,
                    value=inner,
                ),
            ),
        )
        pb = outer.to_pb()
        self.assertEqual(pb.values[0].type, DataType.PROPERTYSET)
        self.assertEqual(
            pb.WhichOneof("timestamp") if hasattr(pb, "timestamp") else "n/a", "n/a"
        )
        # The nested propertyset must be in the propertyset_value field
        nested_pb = pb.values[0].propertyset_value
        self.assertEqual(list(nested_pb.keys), ["unit"])
        self.assertEqual(nested_pb.values[0].string_value, "degC")

        outer2 = PropertySet.from_pb(pb)
        inner_pv = outer2.values[0]
        self.assertEqual(inner_pv.datatype, DataType.PROPERTYSET)
        self.assertIsInstance(inner_pv.value, PropertySet)
        assert isinstance(inner_pv.value, PropertySet)
        self.assertEqual(inner_pv.value.keys, ("unit",))
        self.assertEqual(inner_pv.value.values[0].value, "degC")

    def test_deeply_nested_propertyset(self):
        """PropertySets can be nested arbitrarily deep."""
        level3 = PropertySet(
            keys=("value",),
            values=(PropertyValue(datatype=DataType.DOUBLE, value=1.5),),
        )
        level2 = PropertySet(
            keys=("inner",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=level3),),
        )
        level1 = PropertySet(
            keys=("nested",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=level2),),
        )
        # Roundtrip
        level1_pb = level1.to_pb()
        level1_back = PropertySet.from_pb(level1_pb)

        pv = level1_back.values[0]
        self.assertIsInstance(pv.value, PropertySet)
        assert isinstance(pv.value, PropertySet)
        pv2 = pv.value.values[0]
        self.assertIsInstance(pv2.value, PropertySet)
        assert isinstance(pv2.value, PropertySet)
        pv3 = pv2.value.values[0]
        self.assertAlmostEqual(pv3.value, 1.5, places=6)  # type: ignore[arg-type]

    def test_wrong_type_for_propertyset_raises(self):
        """Passing a non-PropertySet value with PROPERTYSET datatype must raise TypeError."""
        pv = PropertyValue(datatype=DataType.PROPERTYSET, value="wrong type")
        with self.assertRaises(TypeError):
            pv.to_pb()

    def test_propertyset_null(self):
        """A null PROPERTYSET value must encode correctly."""
        pv = PropertyValue(datatype=DataType.PROPERTYSET, is_null=True)
        pb = pv.to_pb()
        self.assertTrue(pb.is_null)
        self.assertEqual(pb.type, DataType.PROPERTYSET)

        pv2 = PropertyValue.from_pb(pb)
        self.assertTrue(pv2.is_null)
        self.assertIsNone(pv2.value)


class TestPropertySetList(unittest.TestCase):
    """Test PropertySetList."""

    def test_empty_propertysetlist(self):
        """An empty PropertySetList must serialize correctly."""
        psl = PropertySetList(propertyset=())
        pb = psl.to_pb()
        self.assertEqual(len(pb.propertyset), 0)

        psl2 = PropertySetList.from_pb(pb)
        self.assertEqual(psl2.propertyset, ())

    def test_single_propertyset_in_list(self):
        """A PropertySetList with one PropertySet must roundtrip."""
        inner = PropertySet(
            keys=("a",),
            values=(PropertyValue(datatype=DataType.INT32, value=1),),
        )
        psl = PropertySetList(propertyset=(inner,))
        psl2 = PropertySetList.from_pb(psl.to_pb())
        self.assertEqual(len(psl2.propertyset), 1)
        self.assertEqual(psl2.propertyset[0].keys, ("a",))
        self.assertEqual(psl2.propertyset[0].values[0].value, 1)

    def test_multiple_propertysets_in_list(self):
        """A PropertySetList preserves order of multiple PropertySets."""
        sets = [
            PropertySet(
                keys=(f"key_{i}",),
                values=(PropertyValue(datatype=DataType.INT32, value=i),),
            )
            for i in range(5)
        ]
        psl = PropertySetList(propertyset=tuple(sets))
        psl2 = PropertySetList.from_pb(psl.to_pb())
        self.assertEqual(len(psl2.propertyset), 5)
        for i, ps in enumerate(psl2.propertyset):
            self.assertEqual(ps.keys, (f"key_{i}",))
            self.assertEqual(ps.values[0].value, i)

    def test_propertysetlist_in_propertyset(self):
        """A PropertyValue with PROPERTYSETLIST datatype must hold a PropertySetList."""
        ps_a = PropertySet(
            keys=("x",),
            values=(PropertyValue(datatype=DataType.INT32, value=10),),
        )
        ps_b = PropertySet(
            keys=("x",),
            values=(PropertyValue(datatype=DataType.INT32, value=20),),
        )
        psl = PropertySetList(propertyset=(ps_a, ps_b))
        outer = PropertySet(
            keys=("readings",),
            values=(
                PropertyValue(
                    datatype=DataType.PROPERTYSETLIST,
                    value=psl,
                ),
            ),
        )
        pb = outer.to_pb()
        self.assertEqual(pb.values[0].type, DataType.PROPERTYSETLIST)
        # verify it uses the propertysets_value field
        self.assertEqual(
            pb.WhichOneof("timestamp") if hasattr(pb, "timestamp") else "n/a", "n/a"
        )
        nested_list_pb = pb.values[0].propertysets_value
        self.assertEqual(len(nested_list_pb.propertyset), 2)

        outer2 = PropertySet.from_pb(pb)
        pv = outer2.values[0]
        self.assertEqual(pv.datatype, DataType.PROPERTYSETLIST)
        self.assertIsInstance(pv.value, PropertySetList)
        assert isinstance(pv.value, PropertySetList)
        self.assertEqual(len(pv.value.propertyset), 2)
        self.assertEqual(pv.value.propertyset[0].values[0].value, 10)
        self.assertEqual(pv.value.propertyset[1].values[0].value, 20)

    def test_wrong_type_for_propertysetlist_raises(self):
        """Passing a non-PropertySetList value with PROPERTYSETLIST datatype must raise TypeError."""
        pv = PropertyValue(
            datatype=DataType.PROPERTYSETLIST, value=PropertySet(keys=(), values=())
        )
        with self.assertRaises(TypeError):
            pv.to_pb()

    def test_propertysetlist_null(self):
        """A null PROPERTYSETLIST value must encode correctly."""
        pv = PropertyValue(datatype=DataType.PROPERTYSETLIST, is_null=True)
        pb = pv.to_pb()
        self.assertTrue(pb.is_null)
        self.assertEqual(pb.type, DataType.PROPERTYSETLIST)

        pv2 = PropertyValue.from_pb(pb)
        self.assertTrue(pv2.is_null)
        self.assertIsNone(pv2.value)

    def test_frozen_dataclass(self):
        """PropertySetList must be immutable (frozen dataclass)."""
        psl = PropertySetList(propertyset=())
        with self.assertRaises((AttributeError, TypeError)):
            psl.propertyset = ()  # type: ignore[misc]


class TestMetricProperties(unittest.TestCase):
    """Test that Metric.properties integrates with the PropertySet implementation."""

    def test_metric_without_properties_is_none(self):
        """A Metric with no properties must have properties=None."""
        metric = Metric(
            timestamp=1000,
            name="test",
            datatype=DataType.INT32,
            value=1,
        )
        self.assertIsNone(metric.properties)
        pb = metric.to_pb(include_dtype=True)
        # The properties field must NOT be set in the protobuf
        self.assertFalse(pb.HasField("properties"))

    def test_metric_without_properties_roundtrip(self):
        """A Metric with no properties must decode without properties."""
        metric = Metric(
            timestamp=1000,
            name="test",
            datatype=DataType.BOOLEAN,
            value=True,
        )
        pb = metric.to_pb(include_dtype=True)
        m2 = Metric.from_pb(pb)
        self.assertIsNone(m2.properties)

    def test_metric_with_simple_properties(self):
        """A Metric with a simple PropertySet must serialize and deserialize."""
        props = PropertySet(
            keys=("eng_unit", "min", "max"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="degC"),
                PropertyValue(datatype=DataType.DOUBLE, value=-40.0),
                PropertyValue(datatype=DataType.DOUBLE, value=125.0),
            ),
        )
        metric = Metric(
            timestamp=1234567890,
            name="temperature",
            datatype=DataType.FLOAT,
            value=25.5,
            properties=props,
        )

        pb = metric.to_pb(include_dtype=True)
        self.assertTrue(pb.HasField("properties"))
        self.assertEqual(list(pb.properties.keys), ["eng_unit", "min", "max"])
        self.assertEqual(pb.properties.values[0].string_value, "degC")
        self.assertAlmostEqual(pb.properties.values[1].double_value, -40.0)
        self.assertAlmostEqual(pb.properties.values[2].double_value, 125.0)

        m2 = Metric.from_pb(pb)
        self.assertIsNotNone(m2.properties)
        assert m2.properties is not None
        self.assertEqual(m2.properties.keys, ("eng_unit", "min", "max"))
        self.assertEqual(m2.properties.values[0].value, "degC")
        self.assertAlmostEqual(m2.properties.values[1].value, -40.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(m2.properties.values[2].value, 125.0)  # type: ignore[arg-type]

    def test_metric_with_null_property_values(self):
        """A Metric can have properties containing null values."""
        props = PropertySet(
            keys=("description",),
            values=(PropertyValue(datatype=DataType.STRING, is_null=True),),
        )
        metric = Metric(
            timestamp=1000,
            name="some_metric",
            datatype=DataType.INT32,
            value=0,
            properties=props,
        )
        pb = metric.to_pb(include_dtype=True)
        m2 = Metric.from_pb(pb)
        assert m2.properties is not None
        self.assertTrue(m2.properties.values[0].is_null)
        self.assertIsNone(m2.properties.values[0].value)

    def test_metric_with_empty_propertyset(self):
        """A Metric with an empty PropertySet must roundtrip correctly."""
        props = PropertySet(keys=(), values=())
        metric = Metric(
            timestamp=1000,
            name="m",
            datatype=DataType.BOOLEAN,
            value=False,
            properties=props,
        )
        pb = metric.to_pb(include_dtype=True)
        self.assertTrue(pb.HasField("properties"))

        m2 = Metric.from_pb(pb)
        assert m2.properties is not None
        self.assertEqual(m2.properties.keys, ())
        self.assertEqual(m2.properties.values, ())

    def test_metric_with_nested_propertyset(self):
        """A Metric can have properties that contain nested PropertySets."""
        inner = PropertySet(
            keys=("source",),
            values=(PropertyValue(datatype=DataType.STRING, value="sensor_A"),),
        )
        outer = PropertySet(
            keys=("nested_props",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=inner),),
        )
        metric = Metric(
            timestamp=999,
            name="complex_metric",
            datatype=DataType.DOUBLE,
            value=42.0,
            properties=outer,
        )
        pb = metric.to_pb(include_dtype=True)
        m2 = Metric.from_pb(pb)
        assert m2.properties is not None
        nested_pv = m2.properties.values[0]
        self.assertEqual(nested_pv.datatype, DataType.PROPERTYSET)
        self.assertIsInstance(nested_pv.value, PropertySet)
        assert isinstance(nested_pv.value, PropertySet)
        self.assertEqual(nested_pv.value.values[0].value, "sensor_A")

    def test_metric_with_propertysetlist_in_properties(self):
        """A Metric can have properties that contain a PropertySetList."""
        ps1 = PropertySet(
            keys=("reading",),
            values=(PropertyValue(datatype=DataType.DOUBLE, value=1.0),),
        )
        ps2 = PropertySet(
            keys=("reading",),
            values=(PropertyValue(datatype=DataType.DOUBLE, value=2.0),),
        )
        psl = PropertySetList(propertyset=(ps1, ps2))
        props = PropertySet(
            keys=("history",),
            values=(PropertyValue(datatype=DataType.PROPERTYSETLIST, value=psl),),
        )
        metric = Metric(
            timestamp=888,
            name="m_with_list",
            datatype=DataType.INT32,
            value=0,
            properties=props,
        )
        pb = metric.to_pb(include_dtype=True)
        m2 = Metric.from_pb(pb)
        assert m2.properties is not None
        history_pv = m2.properties.values[0]
        self.assertEqual(history_pv.datatype, DataType.PROPERTYSETLIST)
        self.assertIsInstance(history_pv.value, PropertySetList)
        assert isinstance(history_pv.value, PropertySetList)
        self.assertEqual(len(history_pv.value.propertyset), 2)

    def test_metric_properties_independent_of_metadata(self):
        """properties and metadata can coexist on the same Metric."""
        from pysparkplug._metadata import Metadata

        props = PropertySet(
            keys=("unit",),
            values=(PropertyValue(datatype=DataType.STRING, value="rpm"),),
        )
        metadata = Metadata(description="shaft speed")
        metric = Metric(
            timestamp=1000,
            name="shaft_speed",
            datatype=DataType.UINT32,
            value=3000,
            metadata=metadata,
            properties=props,
        )
        pb = metric.to_pb(include_dtype=True)
        m2 = Metric.from_pb(pb)
        self.assertIsNotNone(m2.metadata)
        self.assertIsNotNone(m2.properties)
        assert m2.metadata is not None
        assert m2.properties is not None
        self.assertEqual(m2.metadata.description, "shaft speed")
        self.assertEqual(m2.properties.values[0].value, "rpm")


class TestPropertySetRealWorldScenarios(unittest.TestCase):
    """Test realistic use cases for PropertySet (engineering units, limits, etc.)."""

    def test_engineering_units_scenario(self):
        """Demonstrate attaching engineering unit metadata to a metric."""
        props = PropertySet(
            keys=("Engineering Units", "Tooltip"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="°C"),
                PropertyValue(
                    datatype=DataType.STRING,
                    value="Temperature sensor on heat exchanger inlet",
                ),
            ),
        )
        ps2 = PropertySet.from_pb(props.to_pb())
        self.assertEqual(ps2.keys[0], "Engineering Units")
        self.assertEqual(ps2.values[0].value, "°C")

    def test_min_max_limits_scenario(self):
        """Demonstrate attaching alarm limit metadata to a metric."""
        props = PropertySet(
            keys=("Low", "High", "LowLow", "HighHigh"),
            values=(
                PropertyValue(datatype=DataType.DOUBLE, value=0.0),
                PropertyValue(datatype=DataType.DOUBLE, value=100.0),
                PropertyValue(datatype=DataType.DOUBLE, value=-10.0),
                PropertyValue(datatype=DataType.DOUBLE, value=110.0),
            ),
        )
        ps2 = PropertySet.from_pb(props.to_pb())
        self.assertAlmostEqual(ps2.values[0].value, 0.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(ps2.values[1].value, 100.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(ps2.values[2].value, -10.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(ps2.values[3].value, 110.0)  # type: ignore[arg-type]

    def test_boolean_property_flags(self):
        """PropertySet can store boolean flags as metadata."""
        props = PropertySet(
            keys=("Writeable", "Visible"),
            values=(
                PropertyValue(datatype=DataType.BOOLEAN, value=True),
                PropertyValue(datatype=DataType.BOOLEAN, value=False),
            ),
        )
        ps2 = PropertySet.from_pb(props.to_pb())
        self.assertIs(ps2.values[0].value, True)
        self.assertIs(ps2.values[1].value, False)

    def test_large_propertyset(self):
        """PropertySet with many properties must still round-trip correctly."""
        n = 100
        keys = tuple(f"prop_{i}" for i in range(n))
        values = tuple(
            PropertyValue(datatype=DataType.INT32, value=i) for i in range(n)
        )
        ps = PropertySet(keys=keys, values=values)
        ps2 = PropertySet.from_pb(ps.to_pb())
        self.assertEqual(len(ps2.keys), n)
        for i in range(n):
            self.assertEqual(ps2.values[i].value, i)


class TestPropertyValueImmutability(unittest.TestCase):
    """Test that PropertyValue is a frozen dataclass."""

    def test_frozen(self):
        pv = PropertyValue(datatype=DataType.INT32, value=1)
        with self.assertRaises((AttributeError, TypeError)):
            pv.value = 2  # type: ignore[misc]

    def test_equality(self):
        pv1 = PropertyValue(datatype=DataType.INT32, value=42)
        pv2 = PropertyValue(datatype=DataType.INT32, value=42)
        self.assertEqual(pv1, pv2)

    def test_inequality_different_value(self):
        pv1 = PropertyValue(datatype=DataType.INT32, value=42)
        pv2 = PropertyValue(datatype=DataType.INT32, value=99)
        self.assertNotEqual(pv1, pv2)

    def test_inequality_different_datatype(self):
        pv1 = PropertyValue(datatype=DataType.INT32, value=42)
        pv2 = PropertyValue(datatype=DataType.INT64, value=42)
        self.assertNotEqual(pv1, pv2)


class TestQualityCode(unittest.TestCase):
    """Test Quality Code support per Sparkplug B spec section 6.4.9.

    [tck-id-payloads-propertyset-quality-value-type] The type of the Property
    Value MUST be a value of 3 which represents a Signed 32-bit Integer.

    [tck-id-payloads-propertyset-quality-value-value] The value of the Property
    Value MUST be an int_value and be one of the valid quality codes of 0, 192,
    or 500.
    """

    def test_quality_key_constant(self):
        """QUALITY_KEY must be the string 'Quality'."""
        self.assertEqual(QUALITY_KEY, "Quality")

    def test_quality_code_values(self):
        """BAD=0, GOOD=192, STALE=500 per spec."""
        self.assertEqual(QualityCode.BAD, 0)
        self.assertEqual(QualityCode.GOOD, 192)
        self.assertEqual(QualityCode.STALE, 500)

    def test_quality_code_is_int(self):
        """QualityCode members must be integers (IntEnum)."""
        self.assertIsInstance(QualityCode.BAD, int)
        self.assertIsInstance(QualityCode.GOOD, int)
        self.assertIsInstance(QualityCode.STALE, int)

    def _make_quality_pv(self, code: QualityCode) -> PropertyValue:
        """Helper: build a Quality PropertyValue with INT32 type."""
        return PropertyValue(datatype=DataType.INT32, value=int(code))

    def test_quality_type_must_be_int32(self):
        """[tck-id-payloads-propertyset-quality-value-type]
        The type MUST be INT32 (value 3)."""
        for code in QualityCode:
            with self.subTest(code=code):
                pv = self._make_quality_pv(code)
                pb = pv.to_pb()
                self.assertEqual(pb.type, 3)  # DataType.INT32 == 3

    def test_quality_uses_int_value_field(self):
        """[tck-id-payloads-propertyset-quality-value-value]
        The value MUST use the int_value protobuf field."""
        for code in QualityCode:
            with self.subTest(code=code):
                pv = self._make_quality_pv(code)
                pb = pv.to_pb()
                self.assertEqual(pb.WhichOneof("value"), "int_value")

    def test_quality_bad_encodes_to_zero(self):
        """Quality BAD must encode as int_value=0."""
        pb = self._make_quality_pv(QualityCode.BAD).to_pb()
        self.assertEqual(pb.int_value, 0)

    def test_quality_good_encodes_to_192(self):
        """Quality GOOD must encode as int_value=192."""
        pb = self._make_quality_pv(QualityCode.GOOD).to_pb()
        self.assertEqual(pb.int_value, 192)

    def test_quality_stale_encodes_to_500(self):
        """Quality STALE must encode as int_value=500."""
        pb = self._make_quality_pv(QualityCode.STALE).to_pb()
        self.assertEqual(pb.int_value, 500)

    def test_quality_roundtrip(self):
        """Quality PropertyValues must round-trip correctly."""
        for code in QualityCode:
            with self.subTest(code=code):
                pv = self._make_quality_pv(code)
                pb = pv.to_pb()
                pv2 = PropertyValue.from_pb(pb)
                self.assertEqual(pv2.datatype, DataType.INT32)
                self.assertEqual(pv2.value, int(code))

    def test_quality_in_propertyset(self):
        """A PropertySet with a Quality key must encode and decode correctly."""
        props = PropertySet(
            keys=(QUALITY_KEY,),
            values=(self._make_quality_pv(QualityCode.GOOD),),
        )
        pb = props.to_pb()
        self.assertEqual(list(pb.keys), [QUALITY_KEY])
        self.assertEqual(pb.values[0].int_value, 192)
        self.assertEqual(pb.values[0].type, DataType.INT32)

        props2 = PropertySet.from_pb(pb)
        self.assertEqual(props2.keys[0], QUALITY_KEY)
        self.assertEqual(props2.values[0].value, 192)

    def test_quality_attached_to_metric(self):
        """A Metric can carry a Quality property via its properties field."""
        props = PropertySet(
            keys=(QUALITY_KEY,),
            values=(self._make_quality_pv(QualityCode.BAD),),
        )
        metric = Metric(
            timestamp=1000,
            name="pressure",
            datatype=DataType.FLOAT,
            value=3.14,
            properties=props,
        )
        pb = metric.to_pb(include_dtype=True)
        self.assertTrue(pb.HasField("properties"))
        self.assertEqual(list(pb.properties.keys), [QUALITY_KEY])
        self.assertEqual(pb.properties.values[0].int_value, 0)

        m2 = Metric.from_pb(pb)
        assert m2.properties is not None
        self.assertEqual(m2.properties.keys[0], QUALITY_KEY)
        self.assertEqual(m2.properties.values[0].value, 0)

    def test_quality_only_required_when_not_good(self):
        """Spec: Quality is optional and only required if not GOOD.
        A Metric without Quality property is implicitly GOOD."""
        metric_no_quality = Metric(
            timestamp=1000,
            name="temp",
            datatype=DataType.FLOAT,
            value=22.5,
        )
        self.assertIsNone(metric_no_quality.properties)

    def test_quality_stale_in_metric_properties(self):
        """A metric can carry a STALE quality code."""
        props = PropertySet(
            keys=(QUALITY_KEY,),
            values=(self._make_quality_pv(QualityCode.STALE),),
        )
        metric = Metric(
            timestamp=2000,
            name="flow",
            datatype=DataType.DOUBLE,
            value=0.0,
            properties=props,
        )
        m2 = Metric.from_pb(metric.to_pb(include_dtype=True))
        assert m2.properties is not None
        self.assertEqual(m2.properties.values[0].value, QualityCode.STALE)

    def test_quality_propertyset_full_usage(self):
        """Demonstrates the canonical Quality PropertySet pattern from the spec:

            Payload
             └── Metrics (Repeated)
                  ├── Name        (e.g., "Temperature")
                  ├── Datatype    (e.g., Float)
                  ├── Value       (e.g., 23.5)
                  └── PropertySet ── [Key: "Quality"] ──> PropertyValue
                                                              (Type: Int32,
                                                               Value: 0/192/500)

        All three quality codes (BAD=0, GOOD=192, STALE=500) are tested via a
        full payload bytes roundtrip so both encode and decode are exercised in
        a single test.
        """
        from pysparkplug._payload import NBirth

        cases: list[tuple[QualityCode, int]] = [
            (QualityCode.BAD, 0),
            (QualityCode.GOOD, 192),
            (QualityCode.STALE, 500),
        ]

        for quality_code, expected_int in cases:
            with self.subTest(quality_code=quality_code):
                # Build the metric exactly as shown in the spec diagram
                metric = Metric(
                    timestamp=1_000,
                    name="Temperature",
                    datatype=DataType.FLOAT,
                    value=23.5,
                    properties=PropertySet(
                        keys=(QUALITY_KEY,),
                        values=(
                            PropertyValue(
                                datatype=DataType.INT32,
                                value=int(quality_code),
                            ),
                        ),
                    ),
                )

                # Encode to full payload bytes, then decode back
                birth = NBirth(timestamp=1_000, seq=0, metrics=(metric,))
                birth2 = NBirth.decode(birth.encode())
                m2 = birth2.metrics[0]

                # Assert the metric fields
                self.assertEqual(m2.name, "Temperature")
                self.assertEqual(m2.datatype, DataType.FLOAT)
                self.assertAlmostEqual(m2.value, 23.5, places=4)  # type: ignore[arg-type]

                # Assert the PropertySet structure:
                #   exactly one key ("Quality") with an INT32 value
                assert m2.properties is not None
                self.assertEqual(len(m2.properties.keys), 1)
                self.assertEqual(m2.properties.keys[0], QUALITY_KEY)

                quality_pv = m2.properties.values[0]
                self.assertEqual(quality_pv.datatype, DataType.INT32)
                self.assertFalse(quality_pv.is_null)
                self.assertEqual(quality_pv.value, expected_int)


class TestFullRoundtrip(unittest.TestCase):
    """End-to-end tests that encode to protobuf bytes, deserialize back, and
    assert every field value — confirming both the encode and decode paths work
    together in a single test.

    These tests use the actual wire-format bytes (SerializeToString /
    ParseFromString) so they exercise the full stack, not just the in-memory
    protobuf object layer.
    """

    # ------------------------------------------------------------------
    # Flat PropertySet with every supported scalar type
    # ------------------------------------------------------------------

    def test_flat_propertyset_all_scalar_types(self):  # noqa: PLR0915
        """A PropertySet containing one property for each valid scalar DataType
        must survive a full encode → wire bytes → decode roundtrip with the
        correct values and datatypes on the way back.

        Scalar types covered: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32,
        UINT64, FLOAT, DOUBLE, BOOLEAN, STRING, DATETIME, TEXT, UUID.
        """
        import struct

        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

        # Build the original PropertySet
        original = PropertySet(
            keys=(
                "int8_prop",
                "int16_prop",
                "int32_prop",
                "int64_prop",
                "uint8_prop",
                "uint16_prop",
                "uint32_prop",
                "uint64_prop",
                "float_prop",
                "double_prop",
                "bool_true_prop",
                "bool_false_prop",
                "string_prop",
                "datetime_prop",
                "text_prop",
                "uuid_prop",
            ),
            values=(
                PropertyValue(datatype=DataType.INT8, value=-100),
                PropertyValue(datatype=DataType.INT16, value=-30000),
                PropertyValue(datatype=DataType.INT32, value=-2_000_000),
                PropertyValue(datatype=DataType.INT64, value=-(2**62)),
                PropertyValue(datatype=DataType.UINT8, value=200),
                PropertyValue(datatype=DataType.UINT16, value=60000),
                PropertyValue(datatype=DataType.UINT32, value=3_000_000_000),
                PropertyValue(datatype=DataType.UINT64, value=2**60),
                PropertyValue(datatype=DataType.FLOAT, value=1.5),
                PropertyValue(datatype=DataType.DOUBLE, value=3.141592653589793),
                PropertyValue(datatype=DataType.BOOLEAN, value=True),
                PropertyValue(datatype=DataType.BOOLEAN, value=False),
                PropertyValue(datatype=DataType.STRING, value="hello, world"),
                PropertyValue(datatype=DataType.DATETIME, value=ts),
                PropertyValue(datatype=DataType.TEXT, value="some rich text"),
                PropertyValue(
                    datatype=DataType.UUID, value="550e8400-e29b-41d4-a716-446655440000"
                ),
            ),
        )

        # ----- Encode to wire bytes -----
        pb_bytes = original.to_pb().SerializeToString()
        self.assertGreater(len(pb_bytes), 0, "serialized bytes must be non-empty")

        # ----- Decode from wire bytes -----
        from pysparkplug._protobuf import PropertySet as PB_PropertySet

        pb_back = PB_PropertySet()
        pb_back.ParseFromString(pb_bytes)
        decoded = PropertySet.from_pb(pb_back)

        # ----- Assert structure -----
        self.assertEqual(len(decoded.keys), 16)
        self.assertEqual(len(decoded.values), 16)
        self.assertEqual(decoded.keys, original.keys)

        def val(name: str) -> object:
            idx = decoded.keys.index(name)
            return decoded.values[idx].value

        def dt(name: str) -> DataType:
            idx = decoded.keys.index(name)
            return decoded.values[idx].datatype

        # Signed integers
        self.assertEqual(val("int8_prop"), -100)
        self.assertEqual(dt("int8_prop"), DataType.INT8)
        self.assertEqual(val("int16_prop"), -30000)
        self.assertEqual(dt("int16_prop"), DataType.INT16)
        self.assertEqual(val("int32_prop"), -2_000_000)
        self.assertEqual(dt("int32_prop"), DataType.INT32)
        self.assertEqual(val("int64_prop"), -(2**62))
        self.assertEqual(dt("int64_prop"), DataType.INT64)

        # Unsigned integers
        self.assertEqual(val("uint8_prop"), 200)
        self.assertEqual(dt("uint8_prop"), DataType.UINT8)
        self.assertEqual(val("uint16_prop"), 60000)
        self.assertEqual(dt("uint16_prop"), DataType.UINT16)
        self.assertEqual(val("uint32_prop"), 3_000_000_000)
        self.assertEqual(dt("uint32_prop"), DataType.UINT32)
        self.assertEqual(val("uint64_prop"), 2**60)
        self.assertEqual(dt("uint64_prop"), DataType.UINT64)

        # Floating point — float has single precision, compare at 5 d.p.
        float_expected = struct.unpack("f", struct.pack("f", 1.5))[0]
        self.assertAlmostEqual(val("float_prop"), float_expected, places=5)
        self.assertEqual(dt("float_prop"), DataType.FLOAT)
        self.assertAlmostEqual(val("double_prop"), 3.141592653589793, places=12)  # type: ignore[arg-type]
        self.assertEqual(dt("double_prop"), DataType.DOUBLE)

        # Booleans
        self.assertIs(val("bool_true_prop"), True)
        self.assertEqual(dt("bool_true_prop"), DataType.BOOLEAN)
        self.assertIs(val("bool_false_prop"), False)
        self.assertEqual(dt("bool_false_prop"), DataType.BOOLEAN)

        # Strings
        self.assertEqual(val("string_prop"), "hello, world")
        self.assertEqual(dt("string_prop"), DataType.STRING)
        self.assertEqual(val("text_prop"), "some rich text")
        self.assertEqual(dt("text_prop"), DataType.TEXT)
        self.assertEqual(val("uuid_prop"), "550e8400-e29b-41d4-a716-446655440000")
        self.assertEqual(dt("uuid_prop"), DataType.UUID)

        # Datetime — encoded as ms since epoch
        decoded_dt = val("datetime_prop")
        assert isinstance(decoded_dt, datetime.datetime)
        self.assertEqual(dt("datetime_prop"), DataType.DATETIME)
        self.assertEqual(
            int(decoded_dt.timestamp() * 1000),
            int(ts.timestamp() * 1000),
        )

    # ------------------------------------------------------------------
    # Nested PropertySet (PROPERTYSET type value) — full wire roundtrip
    # ------------------------------------------------------------------

    def test_nested_propertyset_full_roundtrip(self):
        """A PropertySet whose value is itself a PropertySet (type=PROPERTYSET)
        must survive a complete encode → wire bytes → decode roundtrip with all
        inner values intact.
        """
        inner = PropertySet(
            keys=("unit", "scale"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="degC"),
                PropertyValue(datatype=DataType.DOUBLE, value=0.1),
            ),
        )
        outer = PropertySet(
            keys=("value", "meta"),
            values=(
                PropertyValue(datatype=DataType.FLOAT, value=98.6),
                PropertyValue(datatype=DataType.PROPERTYSET, value=inner),
            ),
        )

        # ----- Encode -----
        pb_bytes = outer.to_pb().SerializeToString()

        # ----- Decode from wire -----
        from pysparkplug._protobuf import PropertySet as PB_PropertySet

        pb_back = PB_PropertySet()
        pb_back.ParseFromString(pb_bytes)
        decoded = PropertySet.from_pb(pb_back)

        # ----- Assert outer -----
        self.assertEqual(decoded.keys, ("value", "meta"))
        self.assertAlmostEqual(decoded.values[0].value, 98.6, places=4)  # type: ignore[arg-type]
        self.assertEqual(decoded.values[0].datatype, DataType.FLOAT)

        # ----- Assert nested inner -----
        meta_pv = decoded.values[1]
        self.assertEqual(meta_pv.datatype, DataType.PROPERTYSET)
        self.assertIsInstance(meta_pv.value, PropertySet)
        assert isinstance(meta_pv.value, PropertySet)

        inner_decoded = meta_pv.value
        self.assertEqual(inner_decoded.keys, ("unit", "scale"))
        self.assertEqual(inner_decoded.values[0].datatype, DataType.STRING)
        self.assertEqual(inner_decoded.values[0].value, "degC")
        self.assertEqual(inner_decoded.values[1].datatype, DataType.DOUBLE)
        self.assertAlmostEqual(inner_decoded.values[1].value, 0.1, places=10)  # type: ignore[arg-type]

    def test_deeply_nested_propertyset_full_roundtrip(self):
        """Three levels of nested PropertySet must survive full wire roundtrip."""
        level3 = PropertySet(
            keys=("leaf",),
            values=(PropertyValue(datatype=DataType.INT32, value=42),),
        )
        level2 = PropertySet(
            keys=("l3",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=level3),),
        )
        level1 = PropertySet(
            keys=("l2",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=level2),),
        )

        pb_bytes = level1.to_pb().SerializeToString()

        from pysparkplug._protobuf import PropertySet as PB_PropertySet

        pb_back = PB_PropertySet()
        pb_back.ParseFromString(pb_bytes)
        decoded = PropertySet.from_pb(pb_back)

        pv1 = decoded.values[0]
        self.assertEqual(pv1.datatype, DataType.PROPERTYSET)
        assert isinstance(pv1.value, PropertySet)

        pv2 = pv1.value.values[0]
        self.assertEqual(pv2.datatype, DataType.PROPERTYSET)
        assert isinstance(pv2.value, PropertySet)

        pv3 = pv2.value.values[0]
        self.assertEqual(pv3.datatype, DataType.INT32)
        self.assertEqual(pv3.value, 42)

    # ------------------------------------------------------------------
    # PropertySetList (PROPERTYSETLIST type value) — full wire roundtrip
    # ------------------------------------------------------------------

    def test_propertysetlist_full_roundtrip(self):
        """A PropertySetList with multiple PropertySets (each with different
        scalar types) must survive a complete encode → wire bytes → decode
        roundtrip with all values intact.
        """
        ps1 = PropertySet(
            keys=("name", "value"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="sensor_A"),
                PropertyValue(datatype=DataType.DOUBLE, value=1.23),
            ),
        )
        ps2 = PropertySet(
            keys=("name", "value"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="sensor_B"),
                PropertyValue(datatype=DataType.DOUBLE, value=4.56),
            ),
        )
        ps3 = PropertySet(
            keys=("name", "value"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="sensor_C"),
                PropertyValue(datatype=DataType.DOUBLE, value=7.89),
            ),
        )
        psl = PropertySetList(propertyset=(ps1, ps2, ps3))

        # ----- Encode -----
        pb_bytes = psl.to_pb().SerializeToString()

        # ----- Decode from wire -----
        from pysparkplug._protobuf import PropertySetList as PB_PropertySetList

        pb_back = PB_PropertySetList()
        pb_back.ParseFromString(pb_bytes)
        decoded = PropertySetList.from_pb(pb_back)

        # ----- Assert -----
        self.assertEqual(len(decoded.propertyset), 3)

        names = ["sensor_A", "sensor_B", "sensor_C"]
        values = [1.23, 4.56, 7.89]
        for i, ps in enumerate(decoded.propertyset):
            self.assertEqual(ps.keys, ("name", "value"))
            self.assertEqual(ps.values[0].datatype, DataType.STRING)
            self.assertEqual(ps.values[0].value, names[i])
            self.assertEqual(ps.values[1].datatype, DataType.DOUBLE)
            self.assertAlmostEqual(ps.values[1].value, values[i], places=6)  # type: ignore[arg-type]

    def test_propertysetlist_as_propertyset_value_full_roundtrip(self):
        """A PropertySet containing a PROPERTYSETLIST value must survive a
        complete encode → wire bytes → decode roundtrip.
        """
        psl = PropertySetList(
            propertyset=(
                PropertySet(
                    keys=("reading", "valid"),
                    values=(
                        PropertyValue(datatype=DataType.FLOAT, value=10.0),
                        PropertyValue(datatype=DataType.BOOLEAN, value=True),
                    ),
                ),
                PropertySet(
                    keys=("reading", "valid"),
                    values=(
                        PropertyValue(datatype=DataType.FLOAT, value=20.0),
                        PropertyValue(datatype=DataType.BOOLEAN, value=False),
                    ),
                ),
            )
        )
        outer = PropertySet(
            keys=("samples",),
            values=(PropertyValue(datatype=DataType.PROPERTYSETLIST, value=psl),),
        )

        # ----- Encode -----
        pb_bytes = outer.to_pb().SerializeToString()

        # ----- Decode from wire -----
        from pysparkplug._protobuf import PropertySet as PB_PropertySet

        pb_back = PB_PropertySet()
        pb_back.ParseFromString(pb_bytes)
        decoded = PropertySet.from_pb(pb_back)

        # ----- Assert outer -----
        self.assertEqual(decoded.keys, ("samples",))
        samples_pv = decoded.values[0]
        self.assertEqual(samples_pv.datatype, DataType.PROPERTYSETLIST)
        self.assertIsInstance(samples_pv.value, PropertySetList)
        assert isinstance(samples_pv.value, PropertySetList)

        # ----- Assert inner list -----
        decoded_list = samples_pv.value
        self.assertEqual(len(decoded_list.propertyset), 2)

        ps_0 = decoded_list.propertyset[0]
        self.assertAlmostEqual(ps_0.values[0].value, 10.0, places=5)  # type: ignore[arg-type]
        self.assertIs(ps_0.values[1].value, True)

        ps_1 = decoded_list.propertyset[1]
        self.assertAlmostEqual(ps_1.values[0].value, 20.0, places=5)  # type: ignore[arg-type]
        self.assertIs(ps_1.values[1].value, False)

    # ------------------------------------------------------------------
    # Metric.properties — full wire roundtrip via payload bytes
    # ------------------------------------------------------------------

    def test_metric_with_all_scalar_properties_full_roundtrip(self):
        """A Metric carrying a PropertySet with every scalar type must survive
        the complete encode (to payload bytes) → decode roundtrip.
        """
        from pysparkplug._payload import NBirth

        ts_dt = datetime.datetime(2025, 3, 15, 9, 0, 0, tzinfo=datetime.timezone.utc)
        props = PropertySet(
            keys=(
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float",
                "double",
                "bool_t",
                "bool_f",
                "string",
                "datetime",
                "text",
                "uuid",
            ),
            values=(
                PropertyValue(datatype=DataType.INT8, value=-50),
                PropertyValue(datatype=DataType.INT16, value=1000),
                PropertyValue(datatype=DataType.INT32, value=123456),
                PropertyValue(datatype=DataType.INT64, value=-(2**50)),
                PropertyValue(datatype=DataType.UINT8, value=255),
                PropertyValue(datatype=DataType.UINT16, value=65535),
                PropertyValue(datatype=DataType.UINT32, value=0xDEADBEEF),
                PropertyValue(datatype=DataType.UINT64, value=2**55),
                PropertyValue(datatype=DataType.FLOAT, value=2.5),
                PropertyValue(datatype=DataType.DOUBLE, value=2.718281828),
                PropertyValue(datatype=DataType.BOOLEAN, value=True),
                PropertyValue(datatype=DataType.BOOLEAN, value=False),
                PropertyValue(datatype=DataType.STRING, value="sparkplug"),
                PropertyValue(datatype=DataType.DATETIME, value=ts_dt),
                PropertyValue(datatype=DataType.TEXT, value="rich content"),
                PropertyValue(
                    datatype=DataType.UUID, value="12345678-1234-5678-1234-567812345678"
                ),
            ),
        )
        metric = Metric(
            timestamp=1_000_000,
            name="sensor",
            datatype=DataType.FLOAT,
            value=3.14,
            properties=props,
        )

        # Encode through NBirth payload to actual bytes, then decode back
        birth = NBirth(timestamp=1_000_000, seq=0, metrics=(metric,))
        raw_bytes = birth.encode()
        birth2 = NBirth.decode(raw_bytes)

        m2 = birth2.metrics[0]
        assert m2.properties is not None
        self.assertEqual(len(m2.properties.keys), 16)

        def get(name: str) -> PropertyValue:
            idx = m2.properties.keys.index(name)  # type: ignore[union-attr]
            return m2.properties.values[idx]  # type: ignore[union-attr]

        self.assertEqual(get("int8").value, -50)
        self.assertEqual(get("int8").datatype, DataType.INT8)
        self.assertEqual(get("int16").value, 1000)
        self.assertEqual(get("int32").value, 123456)
        self.assertEqual(get("int64").value, -(2**50))
        self.assertEqual(get("uint8").value, 255)
        self.assertEqual(get("uint16").value, 65535)
        self.assertEqual(get("uint32").value, 0xDEADBEEF)
        self.assertEqual(get("uint64").value, 2**55)
        self.assertAlmostEqual(get("float").value, 2.5, places=5)  # type: ignore[arg-type]
        self.assertAlmostEqual(get("double").value, 2.718281828, places=8)  # type: ignore[arg-type]
        self.assertIs(get("bool_t").value, True)
        self.assertIs(get("bool_f").value, False)
        self.assertEqual(get("string").value, "sparkplug")
        self.assertEqual(get("text").value, "rich content")
        self.assertEqual(get("uuid").value, "12345678-1234-5678-1234-567812345678")
        decoded_dt = get("datetime").value
        assert isinstance(decoded_dt, datetime.datetime)
        self.assertEqual(
            int(decoded_dt.timestamp() * 1000),
            int(ts_dt.timestamp() * 1000),
        )

    def test_metric_with_nested_propertyset_full_roundtrip(self):
        """A Metric with a nested PropertySet in its properties must survive
        the full payload bytes roundtrip.
        """
        from pysparkplug._payload import NBirth

        inner = PropertySet(
            keys=("unit", "min", "max"),
            values=(
                PropertyValue(datatype=DataType.STRING, value="bar"),
                PropertyValue(datatype=DataType.DOUBLE, value=0.0),
                PropertyValue(datatype=DataType.DOUBLE, value=10.0),
            ),
        )
        props = PropertySet(
            keys=("Engineering Units",),
            values=(PropertyValue(datatype=DataType.PROPERTYSET, value=inner),),
        )
        metric = Metric(
            timestamp=500,
            name="pressure",
            datatype=DataType.DOUBLE,
            value=4.5,
            properties=props,
        )
        birth = NBirth(timestamp=500, seq=0, metrics=(metric,))
        birth2 = NBirth.decode(birth.encode())

        m2 = birth2.metrics[0]
        assert m2.properties is not None
        eu_pv = m2.properties.values[0]
        self.assertEqual(eu_pv.datatype, DataType.PROPERTYSET)
        self.assertIsInstance(eu_pv.value, PropertySet)
        assert isinstance(eu_pv.value, PropertySet)
        inner2 = eu_pv.value
        self.assertEqual(inner2.keys, ("unit", "min", "max"))
        self.assertEqual(inner2.values[0].value, "bar")
        self.assertAlmostEqual(inner2.values[1].value, 0.0, places=6)  # type: ignore[arg-type]
        self.assertAlmostEqual(inner2.values[2].value, 10.0, places=6)  # type: ignore[arg-type]

    def test_metric_with_propertysetlist_in_properties_full_roundtrip(self):
        """A Metric carrying a PROPERTYSETLIST property must survive the full
        payload bytes roundtrip.
        """
        from pysparkplug._payload import NBirth

        psl = PropertySetList(
            propertyset=(
                PropertySet(
                    keys=("ts", "v"),
                    values=(
                        PropertyValue(datatype=DataType.UINT64, value=1000),
                        PropertyValue(datatype=DataType.FLOAT, value=1.1),
                    ),
                ),
                PropertySet(
                    keys=("ts", "v"),
                    values=(
                        PropertyValue(datatype=DataType.UINT64, value=2000),
                        PropertyValue(datatype=DataType.FLOAT, value=2.2),
                    ),
                ),
            )
        )
        props = PropertySet(
            keys=("history",),
            values=(PropertyValue(datatype=DataType.PROPERTYSETLIST, value=psl),),
        )
        metric = Metric(
            timestamp=3000,
            name="tag",
            datatype=DataType.FLOAT,
            value=0.0,
            properties=props,
        )
        birth = NBirth(timestamp=3000, seq=0, metrics=(metric,))
        birth2 = NBirth.decode(birth.encode())

        m2 = birth2.metrics[0]
        assert m2.properties is not None
        hist_pv = m2.properties.values[0]
        self.assertEqual(hist_pv.datatype, DataType.PROPERTYSETLIST)
        self.assertIsInstance(hist_pv.value, PropertySetList)
        assert isinstance(hist_pv.value, PropertySetList)
        decoded_list = hist_pv.value
        self.assertEqual(len(decoded_list.propertyset), 2)
        self.assertEqual(decoded_list.propertyset[0].values[0].value, 1000)
        self.assertAlmostEqual(
            decoded_list.propertyset[0].values[1].value,
            1.1,  # type: ignore[arg-type]
            places=4,
        )
        self.assertEqual(decoded_list.propertyset[1].values[0].value, 2000)
        self.assertAlmostEqual(
            decoded_list.propertyset[1].values[1].value,
            2.2,  # type: ignore[arg-type]
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
