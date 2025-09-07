from interfaces.i_reflector import IReflector
from core.logger import setup_logger

logger = setup_logger(__name__)

class SimpleReflector(IReflector):
    def reflect(self, execution_result: str) -> str:
        logger.info(f"Reflection input for execution result: {execution_result}")
        if "error" in execution_result.lower():
            logger.error(f"Reflection: Execution failed.")
            return f"Reflection: Execution failed. Need better planning."
        logger.debug(f"Reflection: Execution succeeded.")
        return f"Reflection: Execution succeeded. All good."
