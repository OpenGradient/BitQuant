"""
Unit tests for OpenGradient AlphaSense tools.
"""
import unittest
import os
from unittest.mock import patch, MagicMock

from langchain_core.tools import BaseTool

# Import directly from the module we're testing
from agent.og_alphasense_tools import (
    create_alphasense_read_workflow_tool,
    create_alphasense_run_model_tool,
    ReadWorkflowInput,
    RunModelInput,  # Import this to use in the test
    create_alphasense_toolkit,
    _ensure_sdk_initialized,
    format_model_output,
    format_workflow_output,
    get_model_input
)
# Import from correct locations
from opengradient.alphasense.types import ToolType
from opengradient.types import InferenceResult

class TestAlphaSenseTools(unittest.TestCase):
    """Tests for the AlphaSense tools."""
    
    @patch('opengradient.init')
    def test_ensure_sdk_initialized(self, mock_init):
        """Test that the SDK initialization works correctly."""
        # Set environment variables
        os.environ["OG_PRIVATE_KEY"] = "test_private_key"
        os.environ["OG_EMAIL"] = "test_email"
        os.environ["OG_PASSWORD"] = "test_password"
        
        # Reset the initialization flag
        import agent.og_alphasense_tools
        agent.og_alphasense_tools._sdk_initialized = False
        
        # Call initialize function
        _ensure_sdk_initialized()
        
        # Verify
        mock_init.assert_called_once_with(
            private_key="test_private_key", 
            email="test_email", 
            password="test_password"
        )
        
        # Should be initialized now
        self.assertTrue(agent.og_alphasense_tools._sdk_initialized)
        
        # Calling again should not re-initialize
        mock_init.reset_mock()
        _ensure_sdk_initialized()
        mock_init.assert_not_called()
    
    def test_ensure_sdk_initialized_missing_key(self):
        """Test that SDK initialization raises error with missing key."""
        # Remove environment variables if they exist
        for var in ["OG_PRIVATE_KEY", "OG_EMAIL", "OG_PASSWORD"]:
            if var in os.environ:
                del os.environ[var]
        
        # Reset the initialization flag
        import agent.og_alphasense_tools
        agent.og_alphasense_tools._sdk_initialized = False
        
        # Verify it raises an error
        with self.assertRaises(ValueError):
            _ensure_sdk_initialized()
    
    def test_get_model_input(self):
        """Test the model input processing function."""
        # Test with a dictionary input
        dict_input = {"key": "value"}
        self.assertEqual(get_model_input(model_input=dict_input), dict_input)
        
        # Test with a list input
        list_input = [[1, 2, 3], [4, 5, 6]]
        expected_output = {"data": list_input}
        self.assertEqual(get_model_input(model_input=list_input), expected_output)
    
    def test_format_model_output(self):
        """Test the model output formatting function."""
        # Test with a dictionary
        mock_result = {"result": "test"}
        self.assertEqual(format_model_output(mock_result), str(mock_result))
        
        # Test with an object that has model_output attribute
        class MockInferenceResult:
            def __init__(self):
                self.model_output = {"prediction": 0.95}
        
        mock_obj = MockInferenceResult()
        self.assertEqual(format_model_output(mock_obj), str(mock_obj.model_output))
    
    def test_format_workflow_output(self):
        """Test the workflow output formatting function."""
        mock_result = {"status": "completed", "data": "test"}
        self.assertEqual(format_workflow_output(mock_result), str(mock_result))

    @patch('agent.og_alphasense_tools._ensure_sdk_initialized')
    @patch('agent.og_alphasense_tools.create_read_workflow_tool')
    def test_create_alphasense_read_workflow_tool(self, mock_create_tool, mock_init):
        """Test that the read workflow tool is created properly."""
        # Setup mock
        mock_tool = MagicMock()
        mock_tool.name = "read_alphasense_workflow"
        mock_tool.description = "Reads results from an AlphaSense workflow execution"
        mock_create_tool.return_value = mock_tool
        
        # Call the function
        tool = create_alphasense_read_workflow_tool()
        
        # Verify SDK was initialized
        mock_init.assert_called_once()
        
        # Verify the tool was created with correct parameters
        mock_create_tool.assert_called_once()
        args, kwargs = mock_create_tool.call_args
        self.assertEqual(kwargs["tool_type"].name, "LANGCHAIN")
        self.assertEqual(kwargs["workflow_contract_address"], "0x71178d64eeF1D4b167bdF393407CB3a70F87a338")
        self.assertEqual(kwargs["tool_name"], "read_alphasense_workflow")
        self.assertTrue("Reads results from an AlphaSense workflow" in kwargs["tool_description"])
        
        # Verify the returned tool
        self.assertEqual(tool, mock_tool)
    
    @patch('agent.og_alphasense_tools._ensure_sdk_initialized')
    @patch('agent.og_alphasense_tools.create_run_model_tool')
    def test_create_alphasense_run_model_tool(self, mock_create_tool, mock_init):
        """Test that the run model tool is created properly."""
        # Setup mock
        mock_tool = MagicMock()
        mock_tool.name = "run_alphasense_model"
        mock_tool.description = "Executes an AlphaSense ML model with the given inputs"
        mock_create_tool.return_value = mock_tool
        
        # Call the function
        tool = create_alphasense_run_model_tool()
        
        # Verify SDK was initialized
        mock_init.assert_called_once()
        
        # Verify the tool was created with correct parameters
        mock_create_tool.assert_called_once()
        
        # Print the actual parameters to debug
        args, kwargs = mock_create_tool.call_args
        print("Actual kwargs:", kwargs.keys())
        
        # Check only the parameters we know exist
        self.assertEqual(kwargs["tool_type"].name, "LANGCHAIN")
        self.assertEqual(kwargs["model_cid"], "QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N")
        self.assertEqual(kwargs["tool_name"], "run_alphasense_model")
        self.assertTrue("Executes an AlphaSense ML model" in kwargs["tool_description"])
        
        # Verify the returned tool
        self.assertEqual(tool, mock_tool)
    
    @patch('agent.og_alphasense_tools.create_alphasense_read_workflow_tool')
    @patch('agent.og_alphasense_tools.create_alphasense_run_model_tool')
    def test_create_alphasense_toolkit(self, mock_run_tool, mock_read_tool):
        """Test the toolkit returns the expected tools."""
        # Setup mocks
        mock_read = MagicMock()
        mock_read.name = "read_alphasense_workflow"
        mock_run = MagicMock()
        mock_run.name = "run_alphasense_model"
        
        mock_read_tool.return_value = mock_read
        mock_run_tool.return_value = mock_run
        
        # Call the function
        toolkit = create_alphasense_toolkit()
        
        # Verify
        self.assertEqual(len(toolkit), 2)
        self.assertEqual(toolkit[0].name, "read_alphasense_workflow")
        self.assertEqual(toolkit[1].name, "run_alphasense_model")
        
        # Verify the tool creation functions were called
        mock_read_tool.assert_called_once()
        mock_run_tool.assert_called_once()

    @patch('opengradient.read_workflow_result')
    @patch('opengradient.init')
    def test_read_workflow_tool_invocation(self, mock_init, mock_read_workflow):
        """Test that the read workflow tool invokes the OpenGradient function correctly."""
        # Setup mock
        mock_result = {"status": "completed", "data": "Test result"}
        mock_read_workflow.return_value = mock_result
        
        # Force SDK initialization to be true
        import agent.og_alphasense_tools
        agent.og_alphasense_tools._sdk_initialized = True
        
        # Create tool and run it
        tool = create_alphasense_read_workflow_tool()
        result = tool.invoke({"workflow_contract_address": "0x71178d64eeF1D4b167bdF393407CB3a70F87a338"})
        
        # Verify workflow was read
        mock_read_workflow.assert_called_once_with(contract_address="0x71178d64eeF1D4b167bdF393407CB3a70F87a338")
        self.assertEqual(result, str(mock_result))
    
    @patch('opengradient.infer')
    @patch('opengradient.init')
    def test_run_model_tool_invocation(self, mock_init, mock_infer):
        """Test that the run model tool invokes the OpenGradient function correctly."""
        # Setup mock
        mock_result = {"result": "Test analysis"}
        mock_infer.return_value = (None, mock_result)
        
        # Force SDK initialization to be true
        import agent.og_alphasense_tools
        agent.og_alphasense_tools._sdk_initialized = True
        
        # Input data
        input_data = [[2535.79,2535.79,2505.37,2515.36],[2515.37,2516.37,2497.27,2506.94],
                      [2506.94,2515,2506.35,2508.77],[2508.77,2519,2507.55,2518.79],
                      [2518.79,2522.1,2513.79,2517.92],[2517.92,2521.4,2514.65,2518.13],
                      [2518.13,2525.4,2517.2,2522.6],[2522.59,2528.81,2519.49,2526.12],
                      [2526.12,2530,2524.11,2529.99],[2529.99,2530.66,2525.29,2526]]
        
        # Create tool and run it
        tool = create_alphasense_run_model_tool()
        result = tool.invoke({
            "model_cid": "QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
            "model_input": {"data": input_data}
        })
        
        # Verify the model was called
        mock_infer.assert_called_once()
        args, kwargs = mock_infer.call_args
        self.assertEqual(kwargs["model_cid"], "QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N")
        self.assertEqual(kwargs["model_input"], {"data": input_data})
        
        # Check that the result is properly formatted
        self.assertEqual(result, str(mock_result))

if __name__ == '__main__':
    unittest.main() 