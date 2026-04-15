from pint import UnitRegistry
import logging

ureg = UnitRegistry()

def set_logging_config():
    """
    Create a consistent logging configuration.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt='%m-%d %H:%M:%S',
    )
    logging.getLogger("MDAnalysis").setLevel(logging.WARNING)
    logging.getLogger("pymbar").setLevel(logging.WARNING)