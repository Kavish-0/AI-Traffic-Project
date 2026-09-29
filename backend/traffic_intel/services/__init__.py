"""Service layer (use-cases) between the API routes and the store / configuration / model registry.

Routes stay thin: they validate input, call a service function and return its typed response
model (``api/schemas.py``). No business logic lives in the routes, and nothing here loads torch.
"""
