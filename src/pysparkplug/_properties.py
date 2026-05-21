"""Module defining PropertyValue, PropertySet, and PropertySetList dataclasses"""

import dataclasses
import datetime
import enum
from typing import Optional, Union, cast

from pysparkplug._datatype import DataType
from pysparkplug._protobuf import PropertySet as PB_PropertySet
from pysparkplug._protobuf import PropertySetList as PB_PropertySetList
from pysparkplug._protobuf import PropertyValue as PB_PropertyValue
from pysparkplug._types import MetricValue, Self

__all__ = [
    "QUALITY_KEY",
    "PropertySet",
    "PropertySetList",
    "PropertyValue",
    "QualityCode",
]

# The property key name for quality codes, per the Sparkplug B spec.
QUALITY_KEY: str = "Quality"


class QualityCode(enum.IntEnum):
    """Quality code values for the 'Quality' Sparkplug B property.

    Per the Sparkplug B specification (section 6.4.9), quality codes are
    attached as a ``PropertyValue`` with ``datatype=DataType.INT32`` and the
    property key :data:`QUALITY_KEY`.

    Attributes:
        BAD: The value is bad / unreliable (quality code 0).
        GOOD: The value is good / reliable (quality code 192).
        STALE: The value is stale / not refreshed recently (quality code 500).
    """

    BAD = 0
    GOOD = 192
    STALE = 500


# Datatypes that are valid for use in a PropertyValue, per the Sparkplug B spec.
# Arrays, Bytes, File, Template, and Dataset are not valid PropertyValue types.
_VALID_PROPERTY_VALUE_DATATYPES: frozenset[DataType] = frozenset(
    {
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
        DataType.PROPERTYSET,
        DataType.PROPERTYSETLIST,
    }
)


@dataclasses.dataclass(frozen=True)
class PropertyValue:
    """Class representing a Sparkplug B PropertyValue

    A PropertyValue represents a single key's value within a PropertySet. The
    value type is determined by the ``datatype`` field, which must be one of the
    scalar types (INT8 through UUID), PROPERTYSET, or PROPERTYSETLIST.

    Per the Sparkplug B specification, the ``datatype`` field is required and
    MUST always be present in the encoded form.

    Args:
        datatype:
            the DataType of this property value; must be a valid PropertyValue
            datatype (not an array type, DATASET, BYTES, FILE, or TEMPLATE)
        is_null:
            if True, the value is explicitly null regardless of the ``value``
            field; the encoded form will set is_null=True
        value:
            the actual value; must be compatible with ``datatype``.
            For PROPERTYSET, must be a :class:`PropertySet`.
            For PROPERTYSETLIST, must be a :class:`PropertySetList`.
            For scalar types, must be int, float, bool, or str (or datetime
            for DATETIME). Set to None (with is_null=False) to represent an
            absent value, or use is_null=True for an explicit null.
    """

    datatype: DataType
    is_null: bool = False
    value: Optional[
        Union[
            int, float, bool, str, datetime.datetime, "PropertySet", "PropertySetList"
        ]
    ] = None

    def to_pb(self) -> PB_PropertyValue:  # type: ignore[reportInvalidTypeForm]
        """Returns a Protobuf PropertyValue

        Returns:
            a Protobuf PropertyValue
        """
        pv = PB_PropertyValue()
        pv.type = self.datatype
        if self.is_null or self.value is None:
            pv.is_null = True
        elif self.datatype == DataType.PROPERTYSET:
            if not isinstance(self.value, PropertySet):
                raise TypeError(
                    f"Expected PropertySet for PROPERTYSET datatype, got {type(self.value)}"
                )
            pv.propertyset_value.CopyFrom(self.value.to_pb())
        elif self.datatype == DataType.PROPERTYSETLIST:
            if not isinstance(self.value, PropertySetList):
                raise TypeError(
                    f"Expected PropertySetList for PROPERTYSETLIST datatype, got {type(self.value)}"
                )
            pv.propertysets_value.CopyFrom(self.value.to_pb())
        else:
            setattr(
                pv,
                self.datatype.field,
                self.datatype.encode(cast(MetricValue, self.value)),
            )
        return pv

    @classmethod
    def from_pb(cls, pv: PB_PropertyValue) -> Self:  # type: ignore[reportInvalidTypeForm]
        """Constructs a PropertyValue from a Protobuf PropertyValue

        Args:
            pv: the Protobuf PropertyValue to construct from

        Returns:
            a PropertyValue object
        """
        datatype = DataType(pv.type)
        is_null = pv.is_null
        if is_null:
            return cls(datatype=datatype, is_null=True, value=None)
        value_field = pv.WhichOneof("value")
        if value_field is None:
            return cls(datatype=datatype, is_null=False, value=None)
        if value_field == "propertyset_value":
            value: Optional[
                Union[
                    int,
                    float,
                    bool,
                    str,
                    datetime.datetime,
                    PropertySet,
                    PropertySetList,
                ]
            ] = PropertySet.from_pb(pv.propertyset_value)
        elif value_field == "propertysets_value":
            value = PropertySetList.from_pb(pv.propertysets_value)
        else:
            value = cast(
                Union[int, float, bool, str, datetime.datetime],
                datatype.decode(getattr(pv, value_field)),
            )
        return cls(datatype=datatype, is_null=False, value=value)


@dataclasses.dataclass(frozen=True)
class PropertySet:
    """Class representing a Sparkplug B PropertySet

    A PropertySet is an ordered collection of named property values. It
    consists of parallel ``keys`` and ``values`` tuples: the i-th key in
    ``keys`` corresponds to the i-th value in ``values``.

    PropertySets can be nested: a :class:`PropertyValue` with
    ``datatype=DataType.PROPERTYSET`` holds a ``PropertySet`` as its value,
    enabling hierarchical property structures.

    Per the Sparkplug B specification, the number of keys MUST equal the number
    of values.

    Args:
        keys:
            the property names; must have the same length as ``values``
        values:
            the property values; must have the same length as ``keys``
    """

    keys: tuple[str, ...]
    values: tuple[PropertyValue, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError(
                f"PropertySet keys and values must have the same length, "
                f"got {len(self.keys)} keys and {len(self.values)} values"
            )

    def to_pb(self) -> PB_PropertySet:  # type: ignore[reportInvalidTypeForm]
        """Returns a Protobuf PropertySet

        Returns:
            a Protobuf PropertySet
        """
        ps = PB_PropertySet()
        ps.keys.extend(self.keys)
        ps.values.extend(v.to_pb() for v in self.values)
        return ps

    @classmethod
    def from_pb(cls, ps: PB_PropertySet) -> Self:  # type: ignore[reportInvalidTypeForm]
        """Constructs a PropertySet from a Protobuf PropertySet

        Args:
            ps: the Protobuf PropertySet to construct from

        Returns:
            a PropertySet object
        """
        return cls(
            keys=tuple(ps.keys),
            values=tuple(PropertyValue.from_pb(v) for v in ps.values),
        )


@dataclasses.dataclass(frozen=True)
class PropertySetList:
    """Class representing a Sparkplug B PropertySetList

    A PropertySetList is an ordered list of :class:`PropertySet` objects. It is
    used as the value of a :class:`PropertyValue` with
    ``datatype=DataType.PROPERTYSETLIST``.

    Args:
        propertyset:
            the ordered tuple of PropertySet objects in this list
    """

    propertyset: tuple[PropertySet, ...]

    def to_pb(self) -> PB_PropertySetList:  # type: ignore[reportInvalidTypeForm]
        """Returns a Protobuf PropertySetList

        Returns:
            a Protobuf PropertySetList
        """
        psl = PB_PropertySetList()
        psl.propertyset.extend(ps.to_pb() for ps in self.propertyset)
        return psl

    @classmethod
    def from_pb(cls, psl: PB_PropertySetList) -> Self:  # type: ignore[reportInvalidTypeForm]
        """Constructs a PropertySetList from a Protobuf PropertySetList

        Args:
            psl: the Protobuf PropertySetList to construct from

        Returns:
            a PropertySetList object
        """
        return cls(
            propertyset=tuple(PropertySet.from_pb(ps) for ps in psl.propertyset),
        )
