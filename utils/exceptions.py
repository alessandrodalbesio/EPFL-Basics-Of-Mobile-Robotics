class CenterButtonPressed(Exception): pass

class NotEnoughMarkers(Exception): pass

class DoNotAccessError(Exception):
    """ This exception is raised when a getter is not implemented and will never be implemented since the data stored shouldn't be accessed directly """
    pass
