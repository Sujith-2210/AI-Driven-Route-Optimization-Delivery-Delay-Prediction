"""
Route Optimization Engine
Uses A* algorithm to find optimal delivery routes considering distance and traffic.
"""

import heapq
import math
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in km between two points on earth."""
    R = 6371  # Radius of earth in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_traffic_multiplier(traffic_level: str) -> float:
    """Get travel time multiplier based on traffic level."""
    multipliers = {
        'Low': 1.0,
        'Medium': 1.3,
        'High': 1.6,
        'Very High': 2.0
    }
    return multipliers.get(traffic_level, 1.0)


class DeliveryNode:
    """Represents a delivery location."""
    
    def __init__(self, delivery_id: str, lat: float, lon: float, 
                 traffic_level: str = 'Medium', time_window: Optional[Tuple[int, int]] = None):
        self.delivery_id = delivery_id
        self.lat = lat
        self.lon = lon
        self.traffic_level = traffic_level
        self.time_window = time_window  # (start_hour, end_hour)
    
    def __repr__(self):
        return f"DeliveryNode({self.delivery_id}, lat={self.lat:.4f}, lon={self.lon:.4f})"


class RouteOptimizer:
    """
    Route Optimization using A* algorithm.
    Finds the shortest path considering distance and traffic conditions.
    """
    
    def __init__(self, avg_speed_kmh: float = 30.0):
        self.avg_speed_kmh = avg_speed_kmh
        self.nodes: Dict[str, DeliveryNode] = {}
    
    def add_node(self, node: DeliveryNode):
        """Add a delivery node to the graph."""
        self.nodes[node.delivery_id] = node
    
    def calculate_travel_time(self, from_node: DeliveryNode, to_node: DeliveryNode) -> float:
        """Calculate travel time in hours considering traffic."""
        distance = haversine_distance(
            from_node.lat, from_node.lon,
            to_node.lat, to_node.lon
        )
        traffic_mult = get_traffic_multiplier(to_node.traffic_level)
        return (distance / self.avg_speed_kmh) * traffic_mult
    
    def calculate_distance(self, from_node: DeliveryNode, to_node: DeliveryNode) -> float:
        """Calculate distance between two nodes."""
        return haversine_distance(
            from_node.lat, from_node.lon,
            to_node.lat, to_node.lon
        )
    
    def a_star_optimize(self, start_id: str, delivery_ids: List[str]) -> Dict:
        """
        Use A* algorithm to find optimal delivery sequence.
        
        Args:
            start_id: Starting location (depot) ID
            delivery_ids: List of delivery IDs to visit
            
        Returns:
            Dict with optimized route, total distance, and estimated time
        """
        if start_id not in self.nodes:
            raise ValueError(f"Start node {start_id} not found")
        
        # For TSP-like problem, use nearest neighbor heuristic with A* refinement
        unvisited = set(delivery_ids)
        route = [start_id]
        current_id = start_id
        total_distance = 0
        total_time = 0
        
        while unvisited:
            current_node = self.nodes[current_id]
            
            # Find nearest unvisited node (greedy approach with traffic consideration)
            best_next = None
            best_cost = float('inf')
            
            for next_id in unvisited:
                next_node = self.nodes[next_id]
                
                # Cost = distance + traffic penalty
                distance = self.calculate_distance(current_node, next_node)
                time = self.calculate_travel_time(current_node, next_node)
                
                # A* heuristic: estimated remaining cost
                # (average distance to remaining unvisited nodes)
                remaining = unvisited - {next_id}
                if remaining:
                    heuristic = np.mean([
                        self.calculate_distance(next_node, self.nodes[r])
                        for r in remaining
                    ])
                else:
                    heuristic = 0
                
                cost = time + heuristic * 0.1  # Weight heuristic
                
                if cost < best_cost:
                    best_cost = cost
                    best_next = next_id
                    best_distance = distance
                    best_time = time
            
            if best_next:
                route.append(best_next)
                total_distance += best_distance
                total_time += best_time
                unvisited.remove(best_next)
                current_id = best_next
        
        return {
            'route': route,
            'total_distance_km': round(total_distance, 2),
            'estimated_time_hours': round(total_time, 2),
            'num_stops': len(route) - 1,
            'route_details': self._get_route_details(route)
        }
    
    def _get_route_details(self, route: List[str]) -> List[Dict]:
        """Get detailed information for each stop in the route."""
        details = []
        for i, node_id in enumerate(route):
            node = self.nodes[node_id]
            details.append({
                'stop_number': i,
                'delivery_id': node_id,
                'lat': node.lat,
                'lon': node.lon,
                'traffic_level': node.traffic_level
            })
        return details
    
    def optimize_from_dataframe(self, df: pd.DataFrame, depot_lat: float, depot_lon: float) -> Dict:
        """
        Optimize route from a DataFrame of deliveries.
        
        Args:
            df: DataFrame with columns: delivery_id, dest_lat, dest_lng, traffic_level
            depot_lat, depot_lon: Starting location coordinates
            
        Returns:
            Optimized route information
        """
        # Add depot
        depot_node = DeliveryNode('DEPOT', depot_lat, depot_lon, 'Low')
        self.add_node(depot_node)
        
        # Add delivery nodes
        delivery_ids = []
        for _, row in df.iterrows():
            node = DeliveryNode(
                delivery_id=str(row['delivery_id']),
                lat=row['dest_lat'],
                lon=row['dest_lng'],
                traffic_level=row.get('traffic_level', 'Medium')
            )
            self.add_node(node)
            delivery_ids.append(str(row['delivery_id']))
        
        # Optimize
        return self.a_star_optimize('DEPOT', delivery_ids)


def optimize_deliveries(deliveries: pd.DataFrame, depot_coords: Tuple[float, float] = None) -> Dict:
    """
    Main function to optimize a set of deliveries.
    
    Args:
        deliveries: DataFrame with delivery information
        depot_coords: (lat, lon) of depot, defaults to centroid of deliveries
        
    Returns:
        Optimized route details
    """
    if depot_coords is None:
        # Use centroid as depot
        depot_coords = (deliveries['dest_lat'].mean(), deliveries['dest_lng'].mean())
    
    optimizer = RouteOptimizer(avg_speed_kmh=25)  # Urban speed
    result = optimizer.optimize_from_dataframe(deliveries, depot_coords[0], depot_coords[1])
    
    return result


# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('prepared_logistics_dataset.csv')
    
    # Take a sample of 10 deliveries for demonstration
    sample = df.head(10)[['delivery_id', 'dest_lat', 'dest_lng', 'traffic_level']]
    
    print("Sample deliveries:")
    print(sample)
    print("\n" + "="*50)
    
    # Optimize route
    result = optimize_deliveries(sample)
    
    print("\nOPTIMIZED ROUTE:")
    print(f"Total stops: {result['num_stops']}")
    print(f"Total distance: {result['total_distance_km']} km")
    print(f"Estimated time: {result['estimated_time_hours']} hours")
    print(f"\nRoute sequence:")
    for stop in result['route_details']:
        print(f"  {stop['stop_number']}: {stop['delivery_id']} ({stop['traffic_level']} traffic)")
