class KnownException(Exception):
    """Raised for everything we know of."""

    pass


class MissingEdgeException(KnownException):
    """Raised when an edge is missing in the scene graph. Shouldn't be our fault."""

    pass


class EmptyQueryException(KnownException):
    """Raised when the query is empty. Might be our fault."""

    pass


class ManyAttrCandidatesException(KnownException):
    """Raised when get_attr is called on a list. Might be our fault."""

    pass


class EmptyChoiceException(KnownException):
    """Raised 'choose' presents weird arguments."""

    pass


class QueryPlaceException(KnownException):
    """Raised when place attribute of scene is queried, which isn't part of the scene graph. Not our fault."""

    pass


class NotInWordnetException(KnownException):
    """Raised when a word is not in the synsets. Shouldn't be our fault."""

    pass


class AmbiguousAnswerException(KnownException):
    """Raised when answer contains more than one element. Shouldn't be out fault"""

    pass


class NonTrivialCategoryException(KnownException):
    """Raised when same is called with an argument other than type or name."""

    pass

class IncompleteMetadataException(KnownException):
    """Raised when metadata category matchings are missing entries."""

    pass
