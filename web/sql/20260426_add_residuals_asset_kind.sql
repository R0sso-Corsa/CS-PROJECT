USE cs_project;

ALTER TABLE saved_graph_assets
    MODIFY asset_kind ENUM('summary', 'detail', 'residuals') NOT NULL;
