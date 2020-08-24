from abc import ABCMeta, abstractmethod

from shapely.ops import triangulate
from shapely.geometry import MultiPoint


class Polygonizer(metaclass=ABCMeta):
    """
    Takes in a PIL Image and (optionally) a list of shapely points and returns a list of shapely polygons.
    Output should be independent of the order of points.
    """
    @abstractmethod
    def forward(self, image, points, *args, **kwargs):
        pass

    def __call__(self, image, points, *args, **kwargs):
        polygons = self.forward(image, points, *args, **kwargs)
        polygons = self.simplify(polygons)
        return polygons

    @staticmethod
    def simplify(polygons):
        # TODO: Merge small polygons
        return polygons


class DelaunayTriangulator(Polygonizer):
    def __init__(self):
        super().__init__()

    def forward(self, image, points):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(points)
        triangles = triangulate(points)
        return triangles
