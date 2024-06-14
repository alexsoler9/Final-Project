import open3d as o3d

def main():
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud("C:\Users\mew73\Downloads/output.ply")
    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                      zoom = 0.69999999999999996,
                                      lookat = [ 0.11030453917287106, -0.74425805175091198, 1.6534219431861372 ],
                                      front=[ 0.0, -1, -0.5],
                                      up=[-0.0694, -0.9768, 0.2024])
    
if __name__ == "__main__":
    main()
    # To modify focal length change mono/utils/do_test.py line 259
    # To modify focal length change mono/utils/do_test.py line 316
    # To modify focal length change vit.raft5.small.py line 23