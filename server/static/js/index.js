
$(function (){
    var width = 512, height = 512, offset_x = 32, offset_y = 32;
    var row = 16, col = 16;
    var chess_conf = {
        'width': width,
        'height': height,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'row': row,
        'col': col
    }
    var canvas = $('#canvas')[0];
    var ctx = canvas.getContext('2d');
    var cur_player = 1;
    var cur_state = initChessState(row, col);
    var cur_state_id = null;
    var cur_chess_state = 0;

    function drawChessTable(ctx, width, height, offset_x, offset_y, row, col){
        ctx.fillStyle = 'rgb(200,120,38)';
        ctx.fillRect(0,0,width + 2 * offset_x, height + 2 * offset_y);
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.strokeStyle = 'rgb(0,0,0)';
        for(var i=0;i<row;i++){
            ctx.beginPath();
            ctx.moveTo(offset_x, i * (width / (row-1)) + offset_y);
            ctx.lineTo(width + offset_x, i * (width / (row-1)) + offset_y);
            ctx.stroke();
        }
        for(var i=0;i<col;i++){
            ctx.beginPath();
            ctx.moveTo(i * (height / (col-1)) + offset_x, offset_y);
            ctx.lineTo( i * (height / (col-1)) + offset_x, height + offset_y);
            ctx.stroke();
        }
    }

    function drawChessItem(ctx, x, y, player, chess_conf){
        if(player == 1){
            ctx.fillStyle = 'rgb(0,0,0)';
            ctx.strokeStyle = 'rgb(0,0,0)';
        }else{
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.strokeStyle = 'rgb(255,255,255)';
        }
        ctx.beginPath();
        ctx.arc(
            chess_conf.offset_y + y * (chess_conf.height / (chess_conf.col-1)),
            chess_conf.offset_x + x * (chess_conf.width / (chess_conf.row-1)),
            chess_conf.row / 1.25,
            0, 2 * Math.PI
        );
        ctx.fill();
        ctx.stroke();
    }

    function drawChessState(ctx, state, chess_conf){
        drawChessTable(ctx, chess_conf.width, chess_conf.height, chess_conf.offset_x, chess_conf.offset_y, chess_conf.row, chess_conf.col);
        for(var i=0;i<state.length;i++){
            for(var k=0;k<state.length;k++){
                if(state[i][k] != 0){
                    drawChessItem(ctx, i, k, state[i][k], chess_conf);
                }
            }
        }
    }

    function initChessState(row, col){
        var state = [];
        for(var i=0;i<row;i++){
            var state_row = [];
            for(var k=0;k<col;k++){
                state_row.push(0);
            }
            state.push(state_row);
        }
        return state;
    }

    function updateReminderText(cur_player){
        if(cur_chess_state == 0){
            $('#reminder').text(cur_player == 1?'到你了，执子之手，与子偕老，请落子':'不要着急，等一下电脑思考思考');
        }
        if(cur_chess_state == 1){
            $('#reminder').text('恭喜！你赢了！！击败了电脑！');
        }
        if(cur_chess_state == -1){
            $('#reminder').text('很遗憾，你被电脑击败了！！');
        }
        if(cur_chess_state == 2){
            $('#reminder').text('哎呀呀，居然平局了~~~');
        }
    }

    function updateChessState(){
        function is_in_list(list, pair){
            for(var i=0;i<list.length; i++){
                if(list[i][0] == pair[0] && list[i][1] == pair[1]){
                    return true;
                }
            }
            return false;
        }
        var black_list = [], white_list = [];
        for(var i=0;i<row;i++){
            for(var k=0;k<col;k++){
                if(cur_state[i][k] == 1){
                    black_list.push([i, k]);
                }
                if(cur_state[i][k] == -1){
                    white_list.push([i, k]);
                }
            }
        }
        // black
        for(var i=0;i<black_list.length;i++){
            var cur_x = black_list[i][0], cur_y = black_list[i][1];
            var continuous_count = 0;
            for(var x=Math.max(0, cur_x-4); x<Math.min(row, cur_x+4+1); x++){
                if(is_in_list(black_list, [x, cur_y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = 1;
                    return;
                }
            }
            continuous_count = 0;
            for(var y=Math.max(0, cur_y-4); y<Math.min(col, cur_y+4+1); y++){
                if(is_in_list(black_list, [cur_x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = 1;
                    return;
                }
            }
            continuous_count = 0;
            for(var x=Math.max(0, cur_x-4), y=Math.max(0, cur_y-4); x<Math.min(row, cur_x+4+1) && y<Math.min(col, cur_y+4+1); x++, y++){
                if(is_in_list(black_list, [x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = 1;
                    return;
                }
            }
            continuous_count = 0;
            for(var x=cur_x-4, y=cur_y+4; x<cur_x+4+1 && y>cur_y-4-1; x++, y--){
                if(is_in_list(black_list, [x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = 1;
                    return;
                }
            }
        }
        // white
        for(var i=0;i<white_list.length;i++){
            var cur_x = white_list[i][0], cur_y = white_list[i][1];
            var continuous_count = 0;
            for(var x=Math.max(0, cur_x-4); x<Math.min(row, cur_x+4+1); x++){
                if(is_in_list(white_list, [x, cur_y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = -1;
                    return;
                }
            }
            continuous_count = 0;
            for(var y=Math.max(0, cur_y-4); y<Math.min(col, cur_y+4+1); y++){
                if(is_in_list(white_list, [cur_x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = -1;
                    return;
                }
            }
            continuous_count = 0;
            for(var x=Math.max(0, cur_x-4), y=Math.max(0, cur_y-4); x<Math.min(row, cur_x+4+1) && y<Math.min(col, cur_y+4+1); x++, y++){
                if(is_in_list(white_list, [x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = -1;
                    return;
                }
            }
            continuous_count = 0;
            for(var x=cur_x-4, y=cur_y+4; x<cur_x+4+1 && y>cur_y-4-1; x++, y--){
                if(is_in_list(white_list, [x, y])){
                    continuous_count += 1;
                }else{
                    continuous_count = 0;
                }
                if(continuous_count > 4){
                    cur_chess_state = -1;
                    return;
                }
            }
        }
        if(black_list.length + white_list.length >= row * col){
            cur_chess_state = 2;
        }
    }

    function put_state(state, player){
        $.ajax({
            type:'POST',
            url:'/state/put',
            data:JSON.stringify({
                chess_state: state,
                player: player
            }),
            contentType:'application/json',
            dataType:'json',
            success: function (resp){
                console.log(resp);
                if(resp.code == 0){
                    cur_state_id = resp.data.state_id;
                }else{
                    alert('!!服务端异常!!');
                }
            },
            error: function (){
                alert('!!服务端异常!!');
            }
        })
    }

    function get_state(){
        if(cur_state_id != null){
            $.ajax({
                type:'GET',
                url:'/state/get/'+cur_state_id,
                dataType:'json',
                success: function (resp){
                    console.log(resp);
                    if(resp.code == 0){
                        if(resp.data.state == 1){
                            cur_state_id = null;
                            cur_state = resp.data.chess_state;
                            cur_player = -cur_player;
                            drawChessState(ctx, cur_state, chess_conf);
                            updateChessState();
                            updateReminderText(cur_player);
                        }
                    }else{
                        alert('!!服务端异常!!');
                    }
                },
                error: function (){
                    alert('!!服务端异常!!');
                }
            })
        }
    }

    $(canvas).click(function (e){
        var x_pix = Math.floor(e.pageX - e.target.getBoundingClientRect().left) - offset_x;
        var y_pix = Math.floor(e.pageY - e.target.getBoundingClientRect().top) - offset_y;
        var x_pos = 0, y_pos = 0;
        if(x_pix / (width / (row-1)) - Math.floor(x_pix / (width / (row-1))) < 0.5){
            x_pos = Math.floor(x_pix / (width / (row-1)));
        }else{
            x_pos = Math.ceil(x_pix / (width / (row-1)));
        }
        if(y_pix / (height / (col-1)) - Math.floor(y_pix / (height / (col-1))) < 0.5){
            y_pos = Math.floor(y_pix / (height / (col-1)));
        }else{
            y_pos = Math.ceil(y_pix / (height / (col-1)));
        }
        var tmp = y_pos;
        y_pos = x_pos;
        x_pos = tmp;
        if(cur_state[x_pos][y_pos] == 0 && cur_player == 1 && cur_chess_state == 0) {
            cur_state[x_pos][y_pos] = cur_player;
            cur_player = -cur_player;
            drawChessState(ctx, cur_state, chess_conf);
            updateChessState();
            updateReminderText(cur_player);
            if(cur_chess_state == 0){
                put_state(cur_state, cur_player);
            }
        }
    });

    drawChessTable(ctx, width, height, offset_x, offset_y, row, col);
    updateReminderText(cur_player);

    setInterval(function (){
        get_state();
    }, 1000);
});
